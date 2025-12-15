"""Hierarchical objects module. An HierarchicalShape is composed of a low-resolution and a high-resolution shape, with methods to transfer functions between the two resolutions."""

import abc

from geomfum._registry import HierarchicalMeshRegistry, WhichRegistryMixins
from geomfum.basis import EigenBasis


class HierarchicalShape(abc.ABC):
    """Hierarchical shape. An HierarchicalShape is composed of a low-resolution and a high-resolution shape, with methods to transfer functions between the two resolutions.

    Parameters
    ----------
    low : Shape
        Low-resolution shape.
    high : Shape
        High-resolution shape.
    """

    def __init__(self, low, high):
        self.low = low
        self.high = high

    @abc.abstractmethod
    def scalar_low_high(self, scalar):
        """Transfer scalar from low-resolution to high.

        Parameters
        ----------
        scalar : array-like, shape=[..., low.n_vertices]
            Scalar map on the low-resolution shape.

        Returns
        -------
        high_scalar : array-like, shape=[..., high.n_vertices]
            Scalar map on the high-resolution shape.
        """

    def extend_basis(self, set_as_basis=True):
        """Extend basis.

        See section 3.3. of [MBMR2023]_ for details.

        Parameters
        ----------
        set_as_basis : bool
            Whether to set as basis.

        Return
        ------
        vecs : array-like, shape=[high.n_vertices, spectrum_size]
            Eigenvectors.

        References
        ----------
        .. [MBMR2023] Filippo Maggioli, Daniele Baieri, Simone Melzi, and Emanuele Rodolà.
           “ReMatching: Low-Resolution Representations for Scalable Shape
            Correspondence.” arXiv, October 30, 2023.
            https://doi.org/10.48550/arXiv.2305.09274.
        """
        hvecs = self.scalar_low_high(self.low.basis.full_vecs.T).T

        if set_as_basis:
            basis = EigenBasis(self.low.basis.full_vals, hvecs)
            self.high.set_basis(basis)

        return hvecs


class HierarchicalMesh(WhichRegistryMixins, HierarchicalShape):
    """Hierarchical mesh.

    Parameters
    ----------
    low : TriangleMesh
        Low resolution shape.
    high : TriangleMesh
        High resolution shape.
    """

    _Registry = HierarchicalMeshRegistry


class NestedHierarchicalShape:
    """Nested hierachical shape. An object composed of multiple hierarchical shapes from low to high resolution.

    Parameters
    ----------
    hshapes : list[HierarchicalShape]
        Hierarchical shapes from low to high resolution.
    """

    def __init__(self, hshapes):
        self.hshapes = hshapes

    @property
    def shapes(self):
        """Shapes from low to high resolution.

        Remarks
        -------
        shapes : list[Shape]
            List of shapes from low to high resolution.
        """
        return [hshape.low for hshape in self.hshapes] + [self.hshapes[-1].high]

    @property
    def lowest(self):
        """Lowest resolution shape.

        Returns
        -------
        shape : Shape.
        """
        return self.hshapes[0].low

    @property
    def highest(self):
        """Highest resolution shape.

        Returns
        -------
        shape : Shape.
        """
        return self.hshapes[-1].high

    @classmethod
    def from_hierarchical_shape(cls, shape, HierarchicalShape, **kwargs):
        """Create nested from hierarchical.

        Parameters
        ----------
        shape : Shape.
            High-resolution shape.
        HierarchicalShape : HierarchicalShape object
            Class for the mapping between two resolutions.
            Signature: `(high_res_shape, **kwargs).
        kwargs: dict
            Each must be a list with the proper number of resolution levels.
        """
        n_levels = len(kwargs[list(kwargs.keys())[0]])

        hshapes = []
        for n_level in range(n_levels):
            level_kwargs = {}
            for key, value in kwargs.items():
                level_kwargs[key] = value[n_level]

            hshapes.append(HierarchicalShape(shape, **level_kwargs))
            shape = hshapes[-1].low

        hshapes.reverse()
        return cls(hshapes)

    def scalar_low_high(self, scalar, n_levels=None):
        """Transfer scalar from low-resolution to high.

        Parameters
        ----------
        scalar : array-like, shape=[..., low.n_vertices]
            Scalar map on the low-resolution shape.
        n_levels : int
            Number of levels to transfer scalar.
            If ``None`` transfer up to maximum resolution.

        Returns
        -------
        high_scalar : list[array-like], shape=[..., level.n_vertices]
            Scalar map on the shape at corresponding level.
            As many as number of levels.
        """
        n_levels = n_levels or len(self.hshapes)

        scalars = [scalar]
        for _, hshape in zip(range(n_levels), self.hshapes):
            scalars.append(hshape.scalar_low_high(scalars[-1]))

        return scalars

    def extend_basis(self, set_as_basis=True, n_levels=None):
        """Extend basis.

        See section 3.3. of [MBMR2023]_ for details.

        Parameters
        ----------
        set_as_basis : bool
            Whether to set as basis.
        n_levels : int
            Number of levels to transfer scalar.
            If ``None`` transfer up to maximum resolution.

        Return
        ------
        vecs : list[array-like], shape=[level.n_vertices, spectrum_size]
            Eigenvectors.
            As many as number of levels.

        References
        ----------
        .. [MBMR2023] Filippo Maggioli, Daniele Baieri, Simone Melzi, and Emanuele Rodolà.
           “ReMatching: Low-Resolution Representations for Scalable Shape
            Correspondence.” arXiv, October 30, 2023.
            https://doi.org/10.48550/arXiv.2305.09274.
        """
        n_levels = n_levels or len(self.hshapes)

        vecs = [self.hshapes[0].low.basis.full_vecs]
        for _, hshape in zip(range(n_levels), self.hshapes):
            vecs.append(hshape.extend_basis(set_as_basis=set_as_basis))

        return vecs


class NestedHierarchicalMesh(NestedHierarchicalShape):
    """Nested hierachical mesh."""

    @property
    def hmeshes(self):
        """Meshes from low to high resolution.

        Remarks
        -------
        hshapes : list[HierarchicalMesh]
            Hierarchical meshes from low to high resolution.
        """
        return self.hshapes

    @property
    def meshes(self):
        """Meshes from low to high resolution.

        Remarks
        -------
        meshes : list[Mesh]
            List of meshes from low to high resolution.
        """
        return self.shapes

    @property
    def n_vertices(self):
        """Number of vertices at each level.

        Returns
        -------
        n_vertices : list[int]
        """
        return [mesh_.n_vertices for mesh_ in self.meshes]

    @property
    def n_faces(self):
        """Number of faces at each level.

        Returns
        -------
        n_faces : list[int]
        """
        return [mesh_.faces for mesh_ in self.meshes]
