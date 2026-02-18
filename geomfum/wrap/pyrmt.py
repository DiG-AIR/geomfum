"""pyRMT wrapper."""

import gsops.backend as gs
import numpy as np
from PyRMT import RMTMesh

import geomfum.linalg as la
from geomfum.shape.hierarchical import HierarchicalShape
from geomfum.shape.mesh import TriangleMesh


class PyrmtHierarchicalMesh(HierarchicalShape):
    """Hierarchical mesh from PyRMT.

    Based on [MBMR2023]_.

    Parameters
    ----------
    mesh : TriangleMesh
        High-resolution mesh.
    min_n_samples : int
        Minimum number of vertices in low-resolution mesh.

    References
    ----------
    .. [MBMR2023] Filippo Maggioli, Daniele Baieri, Simone Melzi, and Emanuele Rodolà.
        “ReMatching: Low-Resolution Representations for Scalable Shape
        Correspondence.” arXiv, October 30, 2023.
        https://doi.org/10.48550/arXiv.2305.09274.
    """

    def __init__(self, mesh, min_n_samples):
        if min_n_samples > mesh.n_vertices:
            raise ValueError(
                f"Number of samples ({min_n_samples}) is greater than number of"
                f"vertices of high-resolution mesh ({mesh.n_vertices})"
            )

        low = self._remesh(mesh, min_n_samples)

        super().__init__(low=low, high=mesh)

    def _remesh(self, mesh, min_n_samples):
        vertices = gs.to_numpy(mesh.vertices)
        faces = gs.to_numpy(mesh.faces)

        if vertices.dtype != np.float64:
            vertices = vertices.astype(np.float64)

        if not vertices.flags.f_contiguous:
            vertices = np.asfortranarray(vertices, dtype=np.float64)

        if faces.dtype != np.int32:
            faces = faces.astype(np.int32)

        if not faces.flags.f_contiguous:
            faces = np.asfortranarray(faces, dtype=np.int32)

        rhigh = RMTMesh(vertices, faces)
        rhigh.make_manifold()

        rlow = rhigh.remesh(min_n_samples)
        rlow.clean_up()

        self._rhigh = rhigh
        self._rlow = rlow
        self._baryc_map = rlow.baryc_map(vertices)

        return TriangleMesh(gs.array(rlow.vertices), gs.array(rlow.triangles))

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
        return gs.asarray(la.matvecmul(self._baryc_map, scalar))
