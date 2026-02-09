"""Definition of tetrahedral mesh."""

import gsops.backend as gs

from geomfum.io import load_tetrahedral_mesh

from ._base import Shape


class TetrahedralMesh(Shape):
    """Tetrahedral (volumetric) mesh with vertices, tetrahedra, and differential operators.

    Parameters
    ----------
    vertices : array-like, shape=[n_vertices, 3]
        Vertices of the mesh.
    tets : array-like, shape=[n_tets, 4]
        Tetrahedra of the mesh (each row contains 4 vertex indices).
    faces : array-like, shape=[n_faces, 3], optional
        Boundary faces of the mesh.
    """

    def __init__(self, vertices, tets, faces=None):
        super().__init__(shape_type="tetmesh")
        self.vertices = gs.array(vertices)
        self.tets = gs.array(tets)
        self.faces = gs.array(faces) if faces is not None else None

        # Override the surface gradient with the tetrahedral gradient
        from geomfum.operator import TetrahedralGradient

        self.gradient = TetrahedralGradient(self)

        self._tet_volumes = None
        self._vertex_areas = None
        self._tet_faces = None
        self._edges = None
        self._dist_matrix = None
        self.metric = None

    @classmethod
    def from_file(cls, filename):
        """Load tetrahedral mesh from file.

        Parameters
        ----------
        filename : str
            Path to the mesh file (.mesh, .vtk, .vtu, etc.).

        Returns
        -------
        mesh : TetrahedralMesh
            A tetrahedral mesh.
        """
        vertices, tets = load_tetrahedral_mesh(filename)
        vertices = gs.cast(vertices, dtype=gs.float64)
        tets = gs.cast(tets, dtype=gs.int64)
        return cls(vertices, tets)

    @property
    def n_vertices(self):
        """Number of vertices.

        Returns
        -------
        n_vertices : int
        """
        return self.vertices.shape[0]

    @property
    def n_tets(self):
        """Number of tetrahedra.

        Returns
        -------
        n_tets : int
        """
        return self.tets.shape[0]

    @property
    def n_faces(self):
        """Number of boundary faces.

        Returns
        -------
        n_faces : int
        """
        if self.faces is None:
            return 0
        return self.faces.shape[0]

    @property
    def tet_volumes(self):
        """Volume of each tetrahedron.

        Returns
        -------
        tet_volumes : array-like, shape=[n_tets]
            Absolute volume of each tetrahedron.
        """
        if self._tet_volumes is None:
            v0 = self.vertices[self.tets[:, 0]]
            v1 = self.vertices[self.tets[:, 1]]
            v2 = self.vertices[self.tets[:, 2]]
            v3 = self.vertices[self.tets[:, 3]]

            e10 = v1 - v0
            e20 = v2 - v0
            e30 = v3 - v0

            cross = gs.cross(e20, e10)
            self._tet_volumes = gs.abs(gs.einsum("ij,ij->i", cross, e30)) / 6.0

        return self._tet_volumes

    @property
    def vertex_areas(self):
        """Lumped vertex mass (1/4 of adjacent tetrahedron volumes per vertex).

        Analogous to ``vertex_areas`` on TriangleMesh but represents volume
        contribution per vertex.

        Returns
        -------
        vertex_areas : array-like, shape=[n_vertices]
            Mass associated with each vertex.
        """
        if self._vertex_areas is None:
            device = getattr(self.vertices, "device", None)
            vol = self.tet_volumes
            id_vertices = gs.reshape(self.tets, (-1,))
            val = gs.reshape(
                gs.broadcast_to(gs.expand_dims(vol, axis=-1), (self.n_tets, 4)),
                (-1,),
            )
            # TODO: implement device handling of scatter_sum in gsops and use it here instead of converting to cpu
            vertex_mass = gs.to_device(
                gs.scatter_sum_1d(
                    index=gs.to_device(id_vertices, "cpu"), src=gs.to_device(val, "cpu")
                ),
                device,
            )
            self._vertex_areas = vertex_mass / 4.0

        return self._vertex_areas

    @property
    def tet_faces(self):
        """Faces of each tetrahedron.

        For each tetrahedron, returns the 4 triangular faces formed by
        omitting each vertex in turn.

        Returns
        -------
        tet_faces : array-like, shape=[n_tets, 4, 3]
            Vertex indices of each face of each tetrahedron.
        """
        if self._tet_faces is None:
            face_indices = [
                [1, 2, 3],
                [0, 2, 3],
                [0, 1, 3],
                [0, 1, 2],
            ]
            self._tet_faces = gs.stack(
                [self.tets[:, idxs] for idxs in face_indices], axis=1
            )

        return self._tet_faces

    @property
    def edges(self):
        """Unique edges of the tetrahedral mesh.

        Returns
        -------
        edges : array-like, shape=[n_edges, 2]
            Pairs of vertex indices forming edges.
        """
        if self._edges is None:
            # Each tetrahedron has 6 edges: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
            edge_local = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
            all_i = []
            all_j = []
            for i, j in edge_local:
                all_i.append(self.tets[:, i])
                all_j.append(self.tets[:, j])

            ei = gs.concatenate(all_i)
            ej = gs.concatenate(all_j)

            # Build both directions and take unique
            row = gs.concatenate([ei, ej])
            col = gs.concatenate([ej, ei])
            edges = gs.stack([row, col], axis=-1)
            edges = gs.unique(edges, axis=0)
            self._edges = edges[edges[:, 1] > edges[:, 0]]

        return self._edges

    @property
    def dist_matrix(self):
        """Pairwise distances between all vertices using the equipped metric.

        Returns
        -------
        dist_matrix : array-like, shape=[n_vertices, n_vertices]
            Metric distance matrix.
        """
        if self._dist_matrix is None:
            if self.metric is None:
                raise ValueError("Metric is not set.")
            self._dist_matrix = self.metric.dist_matrix()
        return self._dist_matrix

    def equip_with_metric(self, metric):
        """Equip mesh with a distance metric.

        Parameters
        ----------
        metric : class
            A metric class to use for the mesh.
        """
        self.metric = metric(self)
        self._dist_matrix = None
