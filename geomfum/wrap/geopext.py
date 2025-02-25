"""geopext wrapper."""

import geopext
import numpy as np
import scipy

from geomfum.laplacian import BaseLaplacianFinder


class GeopextMeshLaplacianFinder(BaseLaplacianFinder):
    """Algorithm to find the Laplacian of a mesh.

    Parameters
    ----------
    data_struct : str
        Which data structure to use within rust code.
        One of: "corner_table", "half_edge".
    """

    def __init__(self, data_struct="half_edge"):
        available_ds = ("corner_table", "half_edge")
        if data_struct not in available_ds:
            raise ValueError(
                f"Unknown data structure `{data_struct}`. Choose one of the following: {', '.join(available_ds)}"
            )

        self.data_struct = data_struct

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
        stiff_dict, mass_vec = geopext.mesh_laplacian(
            shape.vertices, shape.faces.ravel().astype(np.uintp), self.data_struct
        )

        indices_i = []
        indices_j = []
        values = []
        for (index_i, index_j), value in stiff_dict.items():
            indices_i.append(index_i)
            indices_j.append(index_j)

            values.append(-value)

        stiffness_matrix = scipy.sparse.coo_matrix(
            (values, (indices_i, indices_j)), shape=(shape.n_vertices, shape.n_vertices)
        ).tocsc()

        indices = range(shape.n_vertices)
        mass_matrix = scipy.sparse.coo_matrix(
            (mass_vec, (indices, indices)), shape=(shape.n_vertices, shape.n_vertices)
        ).tocsc()

        return stiffness_matrix, mass_matrix
