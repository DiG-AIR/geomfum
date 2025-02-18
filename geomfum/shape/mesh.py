"""Definition of triangle mesh."""

import numpy as np
import scipy
import torch
from geomfum.io import load_mesh
from geomfum.operator import (
    FaceDivergenceOperator,
    FaceOrientationOperator,
    FaceValuedGradient,
)

from ._base import Shape


class TriangleMesh(Shape):
    """Triangle mesh.

    Parameters
    ----------
    vertices : array-like, shape=[n_vertices, 3]
        Vertices of the mesh.
    faces : array-like, shape=[n_faces, 3]
        Faces of the mesh.
    """

    def __init__(self, vertices, faces):
        super().__init__(is_mesh=True)
        self.vertices = vertices
        self.faces = faces

        self._face_normals = None
        self._face_areas = None
        self._vertex_areas = None

        self._at_init()

    def _at_init(self):
        self.equip_with_operator(
            "face_valued_gradient", FaceValuedGradient.from_registry
        )
        self.equip_with_operator(
            "face_divergence", FaceDivergenceOperator.from_registry
        )
        self.equip_with_operator(
            "face_orientation_operator", FaceOrientationOperator.from_registry
        )

    @classmethod
    def from_file(cls, filename):
        """Instantiate given a file.

        Returns
        -------
        mesh : TriangleMesh
            A triangle mesh.
        """
        vertices, faces = load_mesh(filename)
        return cls(vertices, faces)

    @property
    def n_vertices(self):
        """Number of vertices.

        Returns
        -------
        n_vertices : int
        """
        return self.vertices.shape[0]

    @property
    def n_faces(self):
        """Number of faces.

        Returns
        -------
        n_faces : int
        """
        return self.faces.shape[0]

    @property
    def face_normals(self):
        """Compute face normals of a triangular mesh.

        Returns
        -------
        normals : array-like, shape=[n_faces, 3]
            Normalized per-face normals.
        """
        if self._face_normals is None:
            v1 = self.vertices[self.faces[:, 0]]
            v2 = self.vertices[self.faces[:, 1]]
            v3 = self.vertices[self.faces[:, 2]]

            normals = np.cross(v2 - v1, v3 - v1)
            normals /= np.linalg.norm(normals, axis=1, keepdims=True)

            self._face_normals = normals

        return self._face_normals

    @property
    def face_areas(self):
        """Compute per-face areas.

        Returns
        -------
        face_areas : array-like, shape=[n_faces]
            Per-face areas.
        """
        if self._face_areas is None:
            v1 = self.vertices[self.faces[:, 0]]
            v2 = self.vertices[self.faces[:, 1]]
            v3 = self.vertices[self.faces[:, 2]]
            self._face_areas = 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1), axis=1)

        return self._face_areas

    @property
    def vertex_areas(self):
        """Compute per-vertex areas.

        Area of a vertex, approximated as one third of the sum of the area of its adjacent triangles.

        Returns
        -------
        vertex_areas : array-like, shape=[n_vertices]
            Per-vertex areas.
        """
        if self._vertex_areas is None:
            # THIS IS JUST A TRICK TO BE FASTER THAN NP.ADD.AT
            I = np.concatenate([self.faces[:, 0], self.faces[:, 1], self.faces[:, 2]])
            J = np.zeros_like(I)

            V = np.tile(self.face_areas / 3, 3)

            self._vertex_areas = np.array(
                scipy.sparse.coo_matrix(
                    (V, (I, J)), shape=(self.n_vertices, 1)
                ).todense()
            ).flatten()

        return self._vertex_areas

    def to_torch(self):
        """Convert to torch tensors."""
        self.vertices = torch.tensor(self.vertices)
        self.faces = torch.tensor(self.faces)
    def to_numpy(self):
        """Convert to numpy array."""
        self.vertices = self.vertices.numpy()
        self.faces = self.faces.numpy()