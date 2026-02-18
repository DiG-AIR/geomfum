import random

import gsops.backend as gs
import numpy as np
import pyvista as pv
from polpo.preprocessing import Map
from polpo.preprocessing.mesh.conversion import DataFromPv
from polpo.preprocessing.mesh.decimation import PvDecimate
from polpo.preprocessing.mesh.io import PvReader
from pyFM.mesh import TriMesh

from geomfum.dataset import NotebooksDataset
from geomfum.shape import TriangleMesh


def MeshDecimationPipeline(target_reduction=0.95):
    """Create pipeline to load (decimated) meshes given filenames."""
    steps = (
        [
            PvReader(),
            lambda mesh: mesh
            if isinstance(mesh, pv.PolyData)
            else mesh.extract_surface(),
        ]
        + (
            [PvDecimate(target_reduction=target_reduction)]
            if target_reduction is not None
            else []
        )
        + [DataFromPv() + (lambda x: (gs.cast(x[0], gs.get_default_dtype()), x[1]))]
    )
    return Map(steps, force_iter=True)


def get_meshes_from_indices(indices, target_reduction=0.95):
    dataset = NotebooksDataset()

    _filenames = [dataset.get_filename(index) for index in indices]

    return [
        TriangleMesh(vertices, faces)
        for vertices, faces in MeshDecimationPipeline(
            target_reduction=target_reduction
        )(_filenames)
    ]


def landmark_randomly(shape, n_landmarks=None):
    if n_landmarks is None:
        n_landmarks = random.randint(1, 5)

    return np.random.choice(shape.n_vertices, size=n_landmarks)


class ShapeCollection:
    # defers calls

    def __init__(
        self, keys, target_reduction=0.95, recompute=False, return_duplicate=False
    ):
        # return_duplicate: whether to duplicate shape when get (useful for cmp)
        # TODO: allow different target reductions
        # TODO: may want same key with different reduction (just need to adapt keys)
        self._keys = keys
        self.target_reduction = target_reduction
        self.recompute = recompute
        self.return_duplicate = return_duplicate

        self._shapes = {}
        self._spectrum_kwargs = None
        self._landmark_indices = None

    def keys(self):
        return self._keys

    def set_spectrum_finder(self, **kwargs):
        self._spectrum_kwargs = kwargs

        return self

    def set_landmarks(self, landmark_indices):
        # array or callable(shape)
        # particularly useful for random landmarks
        self._landmark_indices = landmark_indices

        return self

    def get(self, key):
        # TODO: operate on shape
        shape = self._shapes.get(key, None)
        if self.recompute or shape is None:
            shape = get_meshes_from_indices(
                [key], target_reduction=self.target_reduction
            )[0]
            self._shapes[key] = shape

        # TODO: improve logic for second clause?
        if self._spectrum_kwargs is not None and shape.laplacian._basis is None:
            shape.laplacian.find_spectrum(**self._spectrum_kwargs)

        if self._landmark_indices is not None and shape.landmark_indices is None:
            landmark_indices = (
                self._landmark_indices
                if not callable(self._landmark_indices)
                else self._landmark_indices(shape)
            )
            shape.set_landmarks(landmark_indices)

        if self.return_duplicate:
            return (shape, shape)

        return shape


class MeshCollectionWithPyfm(ShapeCollection):
    # defers calls

    def get(self, key):
        shape, pyfm_shape = self._shapes.get(key, (None, None))
        if self.recompute or shape is None:
            shape = get_meshes_from_indices(
                [key], target_reduction=self.target_reduction
            )[0]
            pyfm_shape = TriMesh(shape.vertices, shape.faces)

            self._shapes[key] = (shape, pyfm_shape)

        # TODO: improve logic for second clause?
        if self._spectrum_kwargs is not None and shape.laplacian._basis is None:
            shape.laplacian.find_spectrum(**self._spectrum_kwargs)

            # TODO: add flag for this?
            # ensures same eigenvalues
            pyfm_shape.eigenvalues = shape.basis.full_vals
            pyfm_shape.eigenvectors = shape.basis.full_vecs

        if self._landmark_indices is not None and shape.landmark_indices is None:
            landmark_indices = (
                self._landmark_indices
                if not callable(self._landmark_indices)
                else self._landmark_indices(shape)
            )
            shape.set_landmarks(landmark_indices)
            pyfm_shape.landmark_indices = landmark_indices

        return shape, pyfm_shape


class _ShapePair:
    def __init__(self, collection):
        self._collection = collection
        self._spectrum_kwargs = {}
        self._landmark_indices = {}

        self.key_a, self.key_b = self._collection.keys()

    def set_spectrum_finder(self, key, **kwargs):
        self._spectrum_kwargs[key] = kwargs

        return self

    def set_landmarks(self, key, landmark_indices):
        # array or callable(shape)
        # particularly useful for random landmarks
        self._landmark_indices[key] = landmark_indices

        return self

    def get(self):
        pair = []
        for key in self._collection.keys():
            if spectrum_kwargs := self._spectrum_kwargs.get(key, {}):
                self._collection.set_spectrum_finder(**spectrum_kwargs)

            if landmark_indices := self._landmark_indices.get(key, {}):
                self._collection.set_landmarks(landmark_indices)

            pair.append(self._collection.get(key))

        return pair


class ShapePair(_ShapePair):
    def __init__(self, key_a, key_b, target_reduction=0.95, recompute=False):
        collection = ShapeCollection(
            [key_a, key_b],
            target_reduction=target_reduction,
            recompute=recompute,
            return_duplicate=False,
        )
        super().__init__(collection)


class ShapePairWithPyFm:
    def __init__(self, key_a, key_b, target_reduction=0.95, recompute=False):
        collection = MeshCollectionWithPyfm(
            [key_a, key_b],
            target_reduction=target_reduction,
            recompute=recompute,
            return_duplicate=False,
        )
        super().__init__(collection)
