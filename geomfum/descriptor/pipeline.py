"""Descriptor pipeline."""

import abc

import numpy as np

import geomfum.linalg as la
from geomfum.descriptor._base import SpectralDescriptor

from ._base import Descriptor, LearnedDescriptor


class Subsampler(abc.ABC):
    """Subsampler."""

    @abc.abstractmethod
    def __call__(self, array):
        pass


class ArangeSubsampler(Subsampler):
    """Subsampler based on arange method."""

    def __init__(self, subsample_step=1, axis=0):
        self.subsample_step = subsample_step
        self.axis = axis

    def __call__(self, array):
        indices = np.arange(0, array.shape[self.axis], self.subsample_step)
        slc = [slice(None)] * array.ndim
        slc[self.axis] = indices

        return array[tuple(slc)]


class Normalizer(abc.ABC):
    """Normalizer."""

    @abc.abstractmethod
    def __call__(self, shape, array):
        pass


class L2InnerNormalizer(Normalizer):
    """L2 inner normalizer."""

    def __call__(self, shape, array):
        coeff = np.sqrt(
            np.einsum(
                "...n,...n->...",
                array,
                la.matvecmul(shape.laplacian.mass_matrix, array),
            ),
        )
        return la.scalarvecmul(1 / coeff, array)


class DescriptorPipeline:
    """Descriptor pipeline."""

    # steps: descriptor, subsampler, normalizer
    def __init__(self, steps):
        self.steps = steps

    def _update_descr(self, current, new):
        if current is None:
            return new
        return np.r_[current, new]

    def apply(self, shape):
        descr = None
        for step in self.steps:
            if isinstance(step, SpectralDescriptor):
                descr = self._update_descr(descr, step(shape.basis))
  
            elif isinstance(step, LearnedDescriptor):
                shape.to_torch()
                descr = self._update_descr(descr, step(shape))
                shape.to_numpy()

            elif isinstance(step, Descriptor):
                descr = self._update_descr(descr, step(shape))
                
            elif isinstance(step, Subsampler):
                descr = step(descr)

            elif isinstance(step, Normalizer):
                descr = step(shape, descr)

            else:
                raise ValueError("Unkown step type.")

        return descr
