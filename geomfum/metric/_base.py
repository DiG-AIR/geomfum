"""Module containing metrics to calculate distances on a Shape."""

import abc

import gsops.backend as gs

from geomfum._registry import HeatDistanceMetricRegistry, MeshWhichRegistryMixins


class Metric(abc.ABC):
    """Abstract base class for distance metrics on shapes.

    Parameters
    ----------
    shape : Shape
        Considered as a manifold.
    """

    def __init__(self, shape):
        self._shape = shape

    @abc.abstractmethod
    def dist(self, point_a, point_b):
        """Distance between points.

        Parameters
        ----------
        point_a : array-like, shape=[...]
            Index Point.
        point_b : array-like, shape=[...]
            Index Point.

        Returns
        -------
        dist : array-like, shape=[...,]
            Distance.
        """


class FinitePointSetMetric(Metric, abc.ABC):
    """Metric supporting distance matrices and source-to-all computations on discrete point sets."""

    @abc.abstractmethod
    def dist_matrix(self):
        """Distances between all the points of a shape.

        Returns
        -------
        dist_matrix : array-like, shape=[n_vertices, n_vertices]
            Distance matrix.
        """

    @abc.abstractmethod
    def dist_from_source(self, source_point):
        """Distances from a source point.

        Parameters
        ----------
        source_point : array-like, shape=[...]
            Index of source point.

        Returns
        -------
        dist : array-like, shape=[...] or list-like[array-like]
            Distance.
        target_point : array-like, shape=[n_targets] or list-like[array-like]
            Target index.
        """


class VertexEuclideanMetric(FinitePointSetMetric):
    """Euclidean distance metric in ambient embedding space."""

    def dist(self, point_a, point_b):
        """Distances between shape vertices.

        Parameters
        ----------
        point_a : array-like, shape=[...]
            Index of source point.
        point_b : array-like, shape=[...]
            Index of target point.

        Returns
        -------
        dist : array-like, shape=[...]
            Distance.
        """
        vertices = self._shape.vertices

        diff = vertices[point_a] - vertices[point_b]
        return gs.linalg.norm(diff, axis=diff.ndim - 1)

    def dist_from_source(self, source_point):
        """Distances from source point.

        Parameters
        ----------
        source_point : array-like, shape=[...]
            Index of source point.

        Returns
        -------
        dist : array-like, shape=[...] or array-like[array-like]
            Distance.
        target_point : array-like, shape=[n_targets] or array-like[array-like]
            Target index.
        """
        vertices = self._shape.vertices

        source_vertices = vertices[source_point]
        if source_vertices.ndim > 1:
            source_vertices = gs.expand_dims(source_vertices, 1)

        diff = source_vertices - vertices

        dist = gs.linalg.norm(diff, axis=diff.ndim - 1)

        target_point = gs.arange(self._shape.n_vertices)
        if diff.ndim > 1:
            target_point = gs.broadcast_to(
                target_point, dist.shape[:-1] + target_point.shape
            )

        return dist, target_point

    def dist_matrix(self):
        """Distances between all shape vertices.

        Returns
        -------
        dist_matrix : array-like, shape=[n_vertices, n_vertices]
            Distance matrix.
        """
        return self.dist_from_source(gs.arange(self._shape.n_vertices))[0]


class HeatDistanceMetric(MeshWhichRegistryMixins):
    """Geodesic distance approximation using the heat method.

    References
    ----------
    .. [CWW2017] Crane, K., Weischedel, C., Wardetzky, M., 2017.
        The heat method for distance computation. Commun. ACM 60, 90–99.
        https://doi.org/10.1145/3131280
    """

    _Registry = HeatDistanceMetricRegistry


class _SingleDispatchMixins:
    """Mixin providing scalar-to-batch dispatch for distance computations."""

    def dist(self, point_a, point_b):
        """Distances between mesh vertices.

        Parameters
        ----------
        point_a : array-like, shape=[...]
            Index of source point.
        point_b : array-like, shape=[...]
            Index of target point.

        Returns
        -------
        dist : array-like, shape=[...,]
            Distance.
        """
        point_a = gs.asarray(point_a)
        point_b = gs.asarray(point_b)

        if point_a.ndim == 0 and point_b.ndim == 0:
            return self._dist_single(point_a, point_b)

        point_a, point_b = gs.broadcast_arrays(point_a, point_b)
        return gs.stack(
            [
                self._dist_single(point_a_, point_b_)
                for point_a_, point_b_ in zip(point_a, point_b)
            ]
        )

    def dist_from_source(self, source_point):
        """Distance between mesh vertices.

        Parameters
        ----------
        source_point : array-like, shape=[...]
            Index of source point.

        Returns
        -------
        dist : array-like, shape=[...,] or list[array-like]
            Distance.
        target_point : array-like, shape=[n_targets,] or list[array-like]
            Target index.
        """
        source_point = gs.asarray(source_point)
        if source_point.ndim == 0:
            return self._dist_from_source_single(source_point)

        out = [
            self._dist_from_source_single(source_index_)
            for source_index_ in source_point
        ]
        return list(zip(*out))

    @abc.abstractmethod
    def _dist_from_source_single(self, source_point):
        pass

    @abc.abstractmethod
    def _dist_single(self, point_a, point_b):
        pass
