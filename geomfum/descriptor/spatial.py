import numpy as np
import scipy
from sklearn.preprocessing import normalize

from geomfum.sample import NearestNeighborsIndexSampler


def linear_compact(x):
    """Linearly decreasing function between 0 and 1."""
    return 1 - x


def poly_compact(x):
    """Polynomial decreasing function between 0 and 1."""
    return 1 - 3 * x**2 + 2 * x**3


def exp_compact(x):
    """Exponential decreasing function between 0 and 1."""
    return np.exp(1 - 1 / (1 - x**2))


class NasikunLocalFunctionsConstructor:
    """

    https://onlinelibrary.wiley.com/doi/full/10.1111/cgf.13496

    Parameters
    ----------
    min_n_samples : int
        Minimum number of samples to target.
        Ignored if ``sampler is not None``.
    """

    # TODO: think about inheritance
    # TODO: check if this is the original source
    # TODO: still a naive implementation

    def __init__(self, min_n_samples=100, sampler=None, local_transform=None):
        if sampler is None:
            sampler = NearestNeighborsIndexSampler(min_n_samples=min_n_samples)

        if local_transform is None:
            local_transform = poly_compact

        self.sampler = sampler
        self.local_transform = local_transform

    def __call__(self, shape):
        # assumes equipped shape

        # TODO: think about how to pass metric
        # TODO: need to implement clone for metric, in case of adaptation
        subsample = self.sampler.sample(shape)

        dists, targets = shape.metric.dist(subsample)

        data = np.hstack(dists)
        row_indices = []
        column_indices = []
        for index, targets_ in enumerate(targets):
            row_indices.extend(targets_.tolist())
            column_indices.extend([index] * len(targets_))

        matrix = scipy.sparse.csr_matrix(
            (data, (row_indices, column_indices)),
            shape=(shape.n_vertices, subsample.size),
        )
        matrix.data = self.local_transform(matrix.data)

        matrix = normalize(matrix, axis=1, norm="l1")

        return matrix
