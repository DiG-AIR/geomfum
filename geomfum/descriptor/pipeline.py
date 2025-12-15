"""Descriptor pipeline."""

import abc

import gsops.backend as gs
import torch

import geomfum.linalg as la
from geomfum.descriptor.learned import LearnedDescriptor

from ._base import Descriptor


class Subsampler(abc.ABC):
    """Abstract base class for array subsampling operations."""

    @abc.abstractmethod
    def __call__(self, array):
        """Subsample array.

        Parameters
        ----------
        array : array-like
            Array to subsample.

        Returns
        -------
        array : array-like
            Subsampled array.
        """


class ArangeSubsampler(Subsampler):
    """Subsampler using uniform stride-based indexing.

    Parameters
    ----------
    subsample_step : int
        Arange step.
    axis : int
        Axis from which to subsample.
    """

    def __init__(self, subsample_step=1, axis=0):
        self.subsample_step = subsample_step
        self.axis = axis

    def __call__(self, array):
        """Subsample array based on arange method.

        Parameters
        ----------
        array : array-like, shape=[..., n, ...]
            Array to subsample.

        Returns
        -------
        array : array-like, shape=[..., d, ...]
            Subsampled array.
        """
        indices = gs.arange(0, array.shape[self.axis], self.subsample_step)
        slc = [slice(None)] * array.ndim
        slc[self.axis] = indices

        return array[tuple(slc)]


class Normalizer(abc.ABC):
    """Abstract base class for descriptor normalization."""

    @abc.abstractmethod
    def __call__(self, shape, array):
        """Normalize array.

        Parameters
        ----------
        shape : Shape
            Shape.
        array : array-like
            Array to normalize.

        Returns
        -------
        array : array-like
            Normalized array.
        """


class L2InnerNormalizer(Normalizer):
    """Normalizer using L2 inner product with mass matrix."""

    def __call__(self, shape, array):
        """Normalize array with respect to L2 inner product.

        Parameters
        ----------
        shape : Shape
            Shape.
        array : array-like, shape=[..., n]
            Array to normalize.

        Returns
        -------
        array : array-like, shape=[..., n]
            Normalized array.
        """
        coeff = gs.sqrt(
            gs.einsum(
                "...n,...n->...",
                array,
                la.matvecmul(shape.laplacian.mass_matrix, array),
            ),
        )
        return la.scalarvecmul(1 / coeff, array)


class DescriptorPipeline:
    """Sequential pipeline for computing and processing shape descriptors.

    Parameters
    ----------
    steps : list or tuple
        Steps to apply.
        Include: descriptor, subsampler, normalizer.
    """

    def __init__(self, steps):
        self.steps = steps

    def _update_descr(self, current, new):
        if current is None:
            return new
        return gs.vstack([current, new])

    def apply(self, shape):
        """Apply descriptor pipeline.

        Parameters
        ----------
        shape : Shape
            Shape to apply pipeline to.

        Returns
        -------
        descr : array-like, shape=[..., n]
            Descriptor.
        """
        descr = None
        for step in self.steps:
            if isinstance(step, Descriptor):
                if isinstance(step, LearnedDescriptor):
                    with torch.no_grad():
                        new = step(shape)
                    descr = self._update_descr(descr, new)
                else:
                    descr = self._update_descr(descr, step(shape))
            elif isinstance(step, Subsampler):
                descr = step(descr)

            elif isinstance(step, Normalizer):
                descr = step(shape, descr)

            else:
                raise ValueError("Unknown step type.")

        return descr
