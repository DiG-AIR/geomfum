"""Igl wrapper."""

import gsops.backend as gs
import igl

from geomfum.laplacian import BaseLaplacianFinder


class IglMeshLaplacianFinder(BaseLaplacianFinder):
    """Algorithm to find the Laplacian of a mesh."""

    def __call__(self, shape):
        """Apply algorithm.

        Parameters
        ----------
        shape : TriangleMesh
            Mesh.

        Returns
        -------
        stiffness_matrix : sparse.csc_matrix, shape=[n_vertices, n_vertices]
            Stiffness matrix.
        mass_matrix : sparse.csc_matrix, shape=[n_vertices, n_vertices]
            Diagonal lumped mass matrix.
        """
        return (
            gs.sparse.from_scipy_csc(-igl.cotmatrix(shape.vertices, shape.faces)),
            gs.sparse.from_scipy_csc(
                igl.massmatrix(shape.vertices, shape.faces, igl.MASSMATRIX_TYPE_VORONOI)
            ),
        )
