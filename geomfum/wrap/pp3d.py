"""potpourri3d wrapper.

https://github.com/nmwsharp/potpourri3d
by Nicholas Sharp.
"""

import gsops.backend as gs
import potpourri3d as pp3d


from geomfum.metric import FinitePointSetMetric
from geomfum.metric._base import _SingleDispatchMixins


class Pp3dHeatDistanceMetric(_SingleDispatchMixins, FinitePointSetMetric):
    """Heat distance metric between vertices of a mesh.

    Parameters
    ----------
    shape : TriangleMesh
        Mesh.
    solver : pp3d.HeatMethodDistanceSolver
        Heat method distance solver class. '  '

    References
    ----------
    .. [CWW2017] Crane, K., Weischedel, C., Wardetzky, M., 2017.
        The heat method for distance computation. Commun. ACM 60, 90–99.
        https://doi.org/10.1145/3131280
    """

    def __init__(self, shape, solver=None):
        super().__init__(shape)
        if solver is None:
            solver = pp3d.MeshHeatMethodDistanceSolver(
                gs.to_numpy(gs.to_device(shape.vertices, "cpu")),
                gs.to_numpy(gs.to_device(shape.faces, "cpu")),
            )

        self.solver = solver

    # TO DO: Dispatch this
    def dist_matrix(self):
        """Distance between mesh vertices.

        Returns
        -------
        dist_matrix : array-like, shape=[n_vertices, n_vertices]
            Distance matrix.

        Notes
        -----
        slow
        """
        dist_mat = []
        for i in range(self._shape.n_vertices):
            dist_mat.append(gs.asarray(self.solver.compute_distance(i)))

        return gs.stack(dist_mat, axis=0)

    def _dist_from_source_single(self, source_point):
        """Distance between mesh vertices.

        Parameters
        ----------
        source_point : array-like, shape=()
            Index of source point.

        Returns
        -------
        dist : array-like, shape=[n_vertices]
            Distance.
        target_point : array-like, shape=[n_vertices,]
            Target index.
        """
        dist = self.solver.compute_distance(source_point.item())

        target_point = gs.arange(self._shape.n_vertices)

        return gs.asarray(dist), target_point

    def _dist_single(self, point_a, point_b):
        """Distance between mesh vertices.

        Parameters
        ----------
        point_a : array-like, shape=()
            Index of source point.
        point_b : array-like, shape=()
            Index of target point.

        Returns
        -------
        dist : numeric
            Distance.
        """
        dist = self.solver.compute_distance(point_a)[point_b]

        return gs.asarray(dist)


class Pp3dMeshHeatDistanceMetric(Pp3dHeatDistanceMetric):
    """Heat distance metric between vertices of a mesh.

    Parameters
    ----------
    shape : TriangleMesh
        Mesh.

    References
    ----------
    cite:: CWW2017
    """

    def __init__(self, shape):
        solver = pp3d.MeshHeatMethodDistanceSolver(
            gs.to_numpy(gs.to_device(shape.vertices, "cpu")),
            gs.to_numpy(gs.to_device(shape.faces, "cpu")),
        )
        super().__init__(shape, solver=solver)


class Pp3dPointSetHeatDistanceMetric(Pp3dHeatDistanceMetric):
    """Heat distance metric between points of a PointCloud.

    Parameters
    ----------
    shape : PointCloud
        Point cloud.

    References
    ----------
    cite:: CWW2017
    """

    def __init__(self, shape):
        solver = pp3d.PointCloudHeatSolver(
            gs.to_numpy(gs.to_device(shape.vertices, "cpu"))
        )
        super().__init__(shape, solver=solver)
