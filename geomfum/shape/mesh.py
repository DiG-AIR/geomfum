"""Definition of triangle mesh."""

import gsops.backend as gs

from geomfum.io import load_mesh
from geomfum.metric import HeatDistanceMetric
from geomfum.operator import (
    FaceDivergenceOperator,
    FaceOrientationOperator,
    FaceValuedGradient,
)
from geomfum.shape.shape_utils import (
    compute_edge_tangent_vectors,
    compute_tangent_frames,
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

    def __init__(
        self,
        vertices,
        faces,
    ):
        super().__init__(is_mesh=True)
        self.vertices = gs.asarray(vertices)
        self.faces = gs.asarray(faces)

        self._edges = None
        self._face_normals = None
        self._face_areas = None
        self._face_area_vectors = None

        self._vertex_areas = None
        self._vertex_normals = None
        self._vertex_tangent_frames = None
        self._edge_tangent_vectors = None
        self._dist_matrix = None
        self.metric = None

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
    def from_file(
        cls,
        filename,
    ):
        """Instantiate given a file.

        Parameters
        ----------
        filename : str
            Path to the mesh file.

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
    def edges(self):
        """Edges of the mesh.

        Returns
        -------
        edges : array-like, shape=[n_edges, 2]
        """
        if self._edges is None:
            vind012 = gs.concatenate(
                [self.faces[:, 0], self.faces[:, 1], self.faces[:, 2]]
            )
            vind120 = gs.concatenate(
                [self.faces[:, 1], self.faces[:, 2], self.faces[:, 0]]
            )
            edges = gs.stack(
                [
                    gs.concatenate([vind012, vind120]),
                    gs.concatenate([vind120, vind012]),
                ],
                axis=-1,
            )
            edges = gs.unique(edges, axis=0)
            self._edges = edges[edges[:, 1] > edges[:, 0]]

        return self._edges

    @property
    def face_vertex_coords(self):
        """Extract vertex coordinates corresponding to each face.

        Returns
        -------
        vertices : array-like, shape=[{n_faces}, n_per_face_vertex, 3]
            Coordinates of the ith vertex of that face.
        """
        return gs.stack(
            [
                self.vertices[self.faces[:, 0]],
                self.vertices[self.faces[:, 1]],
                self.vertices[self.faces[:, 2]],
            ],
            axis=-2,
        )

    @property
    def face_area_vectors(self):
        """Compute face area vectors of a triangular mesh. The face area vector is the vector normal to the face, with a length equal to the area of the face.

        Returns
        -------
        area_vectors : array-like, shape=[n_faces, 3]
            Per-face area vectors.
        """
        if self._face_area_vectors is None:
            face_vertex_coords = self.face_vertex_coords
            self._face_area_vectors = gs.cross(
                face_vertex_coords[:, 1, :] - face_vertex_coords[:, 0, :],
                face_vertex_coords[:, 2, :] - face_vertex_coords[:, 0, :],
            )

        return self._face_area_vectors

    @property
    def face_normals(self):
        """Compute face normals of a triangular mesh.

        Returns
        -------
        normals : array-like, shape=[n_faces, 3]
            Per-face normals.
        """
        if self._face_normals is None:
            face_vertex_coords = self.face_vertex_coords
            self._face_normals = gs.cross(
                face_vertex_coords[:, 1, :] - face_vertex_coords[:, 0, :],
                face_vertex_coords[:, 2, :] - face_vertex_coords[:, 0, :],
            )
            self._face_normals /= gs.linalg.norm(
                self._face_normals, axis=1, keepdims=True
            )

        return self._face_normals

    @property
    def vertex_normals(self):
        """Compute vertex normals of a triangular mesh.

        Returns
        -------
        normals : array-like, shape=[n_vertices, 3]
            Normalized per-vertex normals.
        """
        if self._vertex_normals is None:
            device = getattr(self.vertices, "device", None)

            vind012 = gs.concatenate(
                [self.faces[:, 0], self.faces[:, 1], self.faces[:, 2]]
            )
            zeros = gs.to_device(gs.zeros(len(vind012)), device)

            normals_repeated = gs.vstack([self.face_normals] * 3)
            vertex_normals = gs.to_device(gs.zeros_like(self.vertices), device)
            for c in range(3):
                normals = normals_repeated[:, c]

                vertex_normals[:, c] = gs.asarray(
                    gs.sparse.to_dense(
                        gs.sparse.coo_matrix(
                            gs.stack((vind012, zeros)),
                            normals,
                            shape=(self.n_vertices, 1),
                        )
                    ).flatten()
                )

            vertex_normals = vertex_normals / (
                gs.linalg.norm(vertex_normals, axis=1, keepdims=True) + 1e-12
            )

            self._vertex_normals = vertex_normals

        return self._vertex_normals

    @property
    def face_areas(self):
        """Compute per-face areas.

        Returns
        -------
        face_areas : array-like, shape=[n_faces]
            Per-face areas.
        """
        if self._face_areas is None:
            self._face_areas = 0.5 * gs.linalg.norm(self.face_area_vectors, axis=1)

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
        area = self.face_areas

        id_vertices = gs.broadcast_to(gs.reshape(self.faces, (-1,)), self.n_faces * 3)
        val = gs.reshape(
            gs.broadcast_to(gs.expand_dims(area, axis=-1), (self.n_faces, 3)),
            (-1,),
        )
        incident_areas = gs.scatter_sum_1d(
            index=id_vertices,
            src=val,
        )
        return incident_areas / 3.0

    @property
    def vertex_tangent_frames(self):
        """Compute vertex tangent frame.

        Returns
        -------
        tangent_frame : array-like, shape=[n_vertices, 3, 3]
            Tangent frame of the mesh, where:
            - [n_vertices, 0, :] are the X basis vectors
            - [n_vertices, 1, :] are the Y basis vectors
            - [n_vertices, 2, :] are the vertex normals
        """
        if self._vertex_tangent_frames is None:
            self._vertex_tangent_frames = compute_tangent_frames(
                self.vertices, self.vertex_normals
            )

        return self._vertex_tangent_frames

    @property
    def edge_tangent_vectors(self):
        """Compute edge tangent vectors.

        Returns
        -------
        edge_tangent_vectors : array-like, shape=[n_edges, 2]
            Tangent vectors of the edges, projected onto the local tangent plane.
        """
        if self._edge_tangent_vectors is None:
            edge_tangent_vectors = compute_edge_tangent_vectors(
                self.vertices,
                self.edges,
                self.vertex_tangent_frames,
            )
            self._edge_tangent_vectors = edge_tangent_vectors
        return self._edge_tangent_vectors

    @property  # ToDo
    def dist_matrix(self):
        """Compute metric distance matrix.

        Returns
        -------
        _dist_matrix : array-like, shape=[n_vertices, n_vertices]
            Metric distance matrix.
        """
        if self._dist_matrix is None:
            if self.metric is None:
                raise ValueError("Metric is not set.")
            self._dist_matrix = self.metric.dist_matrix()
        return self._dist_matrix

    def equip_with_metric(self, metric):
        """Set the metric for the mesh.

        Parameters
        ----------
        metric : class
            A metric class to use for the mesh.
        """
        if metric == HeatDistanceMetric:
            self.metric = metric.from_registry(which="pp3d", shape=self)
        else:
            self.metric = metric(self)
        self._dist_matrix = None
