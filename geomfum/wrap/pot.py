"""Python Optimal Trasport wrapper."""

import gsops.backend as gs
import ot

from geomfum.convert import BaseNeighborFinder


class PotSinkhornNeighborFinder(BaseNeighborFinder):
    """Neighbor finder based on Optimal Transport maps computed with Sinkhorn regularization.

    Parameters
    ----------
    n_neighbors : int, default=1
        Number of neighbors to find.
    lambd : float, default=1e-1
        Regularization parameter for Sinkhorn algorithm.
    method : str, default="sinkhorn"
        Method to use for Sinkhorn algorithm.
    max_iter : int, default=100
        Maximum number of iterations for Sinkhorn algorithm.

    References
    ----------
    .. [Cuturi2013] Marco Cuturi. "Sinkhorn Distances: Lightspeed Computation of Optimal Transport."
        Advances in Neural Information Processing Systems (NIPS), 2013.
        http://marcocuturi.net/SI.html
    """

    def __init__(self, n_neighbors=1, lambd=1e-1, method="sinkhorn", max_iter=100):
        super().__init__(n_neighbors=n_neighbors)

        self.lambd = lambd
        self.max_iter = max_iter
        self.method = method

    def __call__(self, X, Y):
        """Find k nearest neighbors using Sinkhorn regularization.

        Parameters
        ----------
        X : array-like, shape=[n_points_x, n_features]
            Query points.
        Y : array-like, shape=[n_points_y, n_features]
            Reference points.

        Returns
        -------
        indices : array-like, shape=[n_points_x, n_neighbors]
            Indices of the nearest neighbors.
        """
        M = gs.exp(-self.lambd * ot.dist(X, Y))

        n, m = M.shape
        a = gs.ones(n) / n
        b = gs.ones(m) / m

        # TODO: implement as sinkhorn solver?
        Gs = ot.sinkhorn(a, b, M, self.lambd, self.method, self.max_iter)

        indices = gs.argsort(Gs, axis=1)[:, : self.n_neighbors]

        return indices
