"""Definition of point cloud."""

from ._base import Shape


class PointCloud(Shape):
    """Point cloud.

    Parameters
    ----------
    vertices : array-like, shape=[n_vertices, 3]
        Vertices of the point cloud.
    """

    def __init__(self, vertices):
        super().__init__(is_mesh=False)
        self.vertices = vertices

    @property
    def n_points(self):
        """Number of vertices.

        Returns
        -------
        n_vertices : int
        """
        return self.vertices.shape[0]
