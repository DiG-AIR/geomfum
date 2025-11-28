"""robust_laplacian wrapper."""

import gsops.backend as gs
import robust_laplacian

from geomfum.laplacian import BaseLaplacianFinder


class RobustMeshLaplacianFinder(BaseLaplacianFinder):
    """Algorithm to find the Laplacian of a mesh.

    Parameters
    ----------
    mollify_factor : float
        Amount of intrinsic mollification to perform.
    """

    def __init__(self, mollify_factor=1e-5):
        self.mollify_factor = mollify_factor

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
        stiffness_matrix, mass_matrix = robust_laplacian.mesh_laplacian(
            gs.to_numpy(gs.to_device(shape.vertices, "cpu")),
            gs.to_numpy(gs.to_device(shape.faces, "cpu")),
            mollify_factor=self.mollify_factor,
        )

        return gs.sparse.from_scipy_csc(stiffness_matrix), gs.sparse.from_scipy_csc(
            mass_matrix
        )


class RobustPointCloudLaplacianFinder(BaseLaplacianFinder):
    """Algorithm to find the Laplacian of a point cloud.

    Parameters
    ----------
    mollify_factor : float
        Amount of intrinsic mollification to perform.
    n_neighbors : float
        Number of nearest neighbors to use when constructing local triangulations.
    """

    def __init__(self, mollify_factor=1e-5, n_neighbors=30):
        self.mollify_factor = mollify_factor
        self.n_neighbors = n_neighbors

    def __call__(self, shape):
        """Apply algorithm.

        Parameters
        ----------
        shape : PointCloud
            Point cloud.

        Returns
        -------
        stiffness_matrix : scipy.sparse.csc_matrix, shape=[n_vertices, n_vertices]
            "Weak" Laplace matrix.
        mass_matrix : scipy.sparse.csc_matrix, shape=[n_vertices, n_vertices]
            Diagonal lumped mass matrix.
        """
        stiffness_matrix, mass_matrix = robust_laplacian.point_cloud_laplacian(
            gs.to_numpy(gs.to_device(shape.vertices, "cpu")),
            mollify_factor=self.mollify_factor,
            n_neighbors=self.n_neighbors,
        )

        return gs.sparse.from_scipy_csc(stiffness_matrix), gs.sparse.from_scipy_csc(
            mass_matrix
        )
