"""Definition of graph."""

import gsops.backend as gs

from ._base import Shape


class Graph(Shape):
    """Graph shape with adjacency-based connectivity.

    Parameters
    ----------
    adjacency_matrix : sparse matrix, shape=[n_vertices, n_vertices], optional
        Weighted adjacency matrix. Mutually exclusive with
        ``edge_index``/``edge_weights``.
    edge_index : array-like, shape=[2, n_edges], optional
        Edge index array where ``edge_index[0]`` are source nodes
        and ``edge_index[1]`` are target nodes.
    edge_weights : array-like, shape=[n_edges], optional
        Edge weights. Defaults to 1 for all edges when used
        with ``edge_index``.
    vertices : array-like, shape=[n_vertices, d], optional
        Optional vertex positions for visualization.
    """

    def __init__(
        self,
        adjacency_matrix=None,
        edge_index=None,
        edge_weights=None,
        vertices=None,
    ):
        super().__init__(shape_type="graph")

        if adjacency_matrix is not None and edge_index is not None:
            raise ValueError(
                "Provide either `adjacency_matrix` or `edge_index`, not both."
            )

        if adjacency_matrix is not None:
            self._adjacency_matrix = adjacency_matrix
            self._n_vertices = adjacency_matrix.shape[0]
            self._edge_index = None
            self._edge_weights = None
        elif edge_index is not None:
            self._edge_index = gs.asarray(edge_index)
            self._edge_weights = (
                gs.asarray(edge_weights) if edge_weights is not None else None
            )
            self._adjacency_matrix = None
            self._n_vertices = int(gs.max(self._edge_index)) + 1
        else:
            raise ValueError(
                "Provide either `adjacency_matrix` or `edge_index`."
            )

        self.vertices = gs.asarray(vertices) if vertices is not None else None

        self._vertex_areas = None
        self._edges = None

    @property
    def n_vertices(self):
        """Number of vertices.

        Returns
        -------
        n_vertices : int
        """
        return self._n_vertices

    @property
    def adjacency_matrix(self):
        """Weighted adjacency matrix.

        Returns
        -------
        adjacency_matrix : sparse matrix, shape=[n_vertices, n_vertices]
        """
        if self._adjacency_matrix is None:
            edge_index = self._edge_index
            edge_weights = self._edge_weights
            if edge_weights is None:
                edge_weights = gs.ones(edge_index.shape[1])

            self._adjacency_matrix = gs.sparse.csc_matrix(
                edge_index,
                edge_weights,
                shape=(self._n_vertices, self._n_vertices),
                coalesce=True,
            )
        return self._adjacency_matrix

    @property
    def vertex_areas(self):
        """Vertex areas based on node degree.

        For graphs, vertex area is defined as the degree (sum of incident
        edge weights) of each vertex.

        Returns
        -------
        vertex_areas : array-like, shape=[n_vertices]
        """
        if self._vertex_areas is None:
            adj_dense = gs.sparse.to_dense(self.adjacency_matrix)
            self._vertex_areas = gs.sum(adj_dense, axis=1)
        return self._vertex_areas

    @property
    def edges(self):
        """Edges of the graph.

        Returns
        -------
        edges : array-like, shape=[n_edges, 2]
        """
        if self._edges is None:
            if self._edge_index is not None:
                self._edges = self._edge_index.T
            else:
                adj_dense = gs.sparse.to_dense(self.adjacency_matrix)
                row_indices, col_indices = gs.where(adj_dense > 0)
                self._edges = gs.stack([row_indices, col_indices], axis=-1)
        return self._edges
