"""Utility functions for shape computations."""

import gsops.backend as gs


def compute_tangent_frames(vertices, normals):
    """Construct local orthonormal frames at each vertex from normal vectors.

    Parameters
    ----------
    vertices : array-like, shape=[n_vertices, 3]
        Vertex coordinates.
    normals : array-like, shape=[n_vertices, 3]
        Vertex normals.

    Returns
    -------
    frames : array-like, shape=[n_vertices, 3, 3]
        Tangent frames.
    """
    device = getattr(normals, "device", None)
    n_vertices = vertices.shape[0]

    tangent_frame = gs.to_device(gs.zeros((n_vertices, 3, 3)), device=device)
    tangent_frame[:, 2, :] = normals

    basis_cand1 = gs.to_device(gs.tile([1, 0, 0], (n_vertices, 1)), device=device)
    basis_cand2 = gs.to_device(gs.tile([0, 1, 0], (n_vertices, 1)), device=device)

    dot_products = gs.sum(normals * basis_cand1, axis=1, keepdims=True)
    basis_x = gs.where(gs.abs(dot_products) < 0.9, basis_cand1, basis_cand2)

    normal_projections = gs.sum(basis_x * normals, axis=1, keepdims=True) * normals
    basis_x = basis_x - normal_projections

    basis_x_norm = gs.linalg.norm(basis_x, axis=1, keepdims=True)
    basis_x = basis_x / (basis_x_norm + 1e-12)

    basis_y = gs.cross(normals, basis_x)

    tangent_frame[:, 0, :] = basis_x
    tangent_frame[:, 1, :] = basis_y

    return tangent_frame


def compute_edge_tangent_vectors(vertices, edges, tangent_frames):
    """Project edge vectors onto local tangent plane coordinates.

    Parameters
    ----------
    vertices : array-like, shape=[n_vertices, 3]
        Vertex coordinates.
    edges : array-like, shape=[n_edges, 2]
        Edges of the shape.
    tangent_frames : array-like, shape=[n_vertices, 3, 3]
        Tangent frames for each vertex.

    Returns
    -------
    edge_tangent_vectors : array-like, shape=[n_edges, 2]
        Tangent vectors of the edges, projected onto the local tangent plane.
    """
    edge_vecs = vertices[edges[:, 1], :] - vertices[edges[:, 0], :]

    basis_x = tangent_frames[edges[:, 0], 0, :]
    basis_y = tangent_frames[edges[:, 0], 1, :]

    comp_x = gs.sum(edge_vecs * basis_x, axis=1)
    comp_y = gs.sum(edge_vecs * basis_y, axis=1)

    return gs.stack((comp_x, comp_y), axis=-1)


def compute_gradient_matrix_fem(vertices, edges, edge_tangent_vectors):
    """Construct gradient operator using local least-squares approximation.

    Parameters
    ----------
    vertices : array-like, shape=[n_vertices, 3]
        Vertex coordinates.
    edges : array-like, shape=[n_edges, 2]
        Edges of the shape.
    edge_tangent_vectors : array-like, shape=[n_edges, 2]
        Tangent vectors of the edges, projected onto the local tangent plane.

    Returns
    -------
    grad_matrix : array-like, shape=[n_edges, n_vertices]
        Gradient matrix.
    """
    n_vertices = vertices.shape[0]
    outgoing_edges_per_vertex = [[] for _ in range(n_vertices)]
    for edge_index in range(edges.shape[0]):
        tail_ind = edges[edge_index, 0]
        tip_ind = edges[edge_index, 1]
        if tip_ind != tail_ind:
            outgoing_edges_per_vertex[tail_ind].append(edge_index)

    row_inds = []
    col_inds = []
    data_vals = []
    eps_reg = 1e-5

    # For each vertex, fit a local linear function 'f' to its neighbors
    for vertex_idx in range(n_vertices):
        num_neighbors = len(outgoing_edges_per_vertex[vertex_idx])

        if num_neighbors == 0:
            continue

        # Set up the least squares system for the local neighborhood
        lhs_mat = gs.zeros((num_neighbors, 2))  # Edge tangent vectors
        rhs_mat = gs.zeros(
            (num_neighbors, num_neighbors + 1)
        )  # Finite Difference matrix rhs_mat[i,j] = f(j) - f(i)
        lookup_vertices_idx = [vertex_idx]

        # for each row of the rhs_mat, we have the following:
        # - rhs_mat[i, 0] = -f(center) (the value at the center vertex)
        # - rhs_mat[i, i + 1] = +f(neighbor) (the value at the neighbor vertex)
        # - rhs_mat[i, j] = 0 for j != 0, i + 1 (no other values)
        for neighbor_index in range(num_neighbors):
            edge_index = outgoing_edges_per_vertex[vertex_idx][neighbor_index]
            neighbor_vertex_idx = edges[edge_index, 1]
            lookup_vertices_idx.append(neighbor_vertex_idx)

            edge_vec = edge_tangent_vectors[edge_index][:]

            lhs_mat[neighbor_index][:] = edge_vec
            rhs_mat[neighbor_index][0] = -1
            rhs_mat[neighbor_index][neighbor_index + 1] = 1

        # Solve
        lhs_T = lhs_mat.T
        lhs_inv = gs.linalg.inv(lhs_T @ lhs_mat + eps_reg * gs.eye(2)) @ lhs_T
        sol_mat = lhs_inv @ rhs_mat
        sol_coefs = gs.transpose((sol_mat[0, :] + 1j * sol_mat[1, :]))

        for i_neigh in range(num_neighbors + 1):
            i_glob = lookup_vertices_idx[i_neigh]
            row_inds.append(vertex_idx)
            col_inds.append(i_glob)
            data_vals.append(sol_coefs[i_neigh])

    # Build the sparse matrix
    row_inds = gs.asarray(row_inds)
    col_inds = gs.asarray(col_inds)
    data_vals = gs.asarray(data_vals)

    gradient_matrix = gs.sparse.to_csc(
        gs.sparse.coo_matrix(
            gs.stack([row_inds, col_inds]),
            data_vals,
            shape=(n_vertices, n_vertices),
        )
    )

    return gradient_matrix
