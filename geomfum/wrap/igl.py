"""igl wrapper."""

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
        stiffness_matrix : scipy.sparse.csc_matrix, shape=[n_vertices, n_vertices]
            Stiffness matrix.
        mass_matrix : scipy.sparse.csc_matrix, shape=[n_vertices, n_vertices]
            Diagonal lumped mass matrix.
        """
        return (
            -igl.cotmatrix(shape.vertices, shape.faces),
            igl.massmatrix(shape.vertices, shape.faces, igl.MASSMATRIX_TYPE_VORONOI),
        )
