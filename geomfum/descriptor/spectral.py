"""Spectral descriptors."""

import geomstats.backend as gs

import geomfum.backend as xgs
import geomfum.linalg as la
from geomfum._registry import (
    HeatKernelSignatureRegistry,
    LandmarkHeatKernelSignatureRegistry,
    LandmarkWaveKernelSignatureRegistry,
    WaveKernelSignatureRegistry,
    WhichRegistryMixins,
)

from ._base import SpectralDescriptor


def hks_default_domain(shape, n_domain):
    """Compute HKS default domain.

    Parameters
    ----------
    shape : Shape.
        Shape with basis.
    n_domain : int
        Number of time points.

    Returns
    -------
    domain : array-like, shape=[n_domain]
        Time points.
    """
    nonzero_vals = shape.basis.nonzero_vals
    device = getattr(nonzero_vals, "device", None)

    return xgs.to_device(
        xgs.geomspace(
            4 * gs.log(10) / nonzero_vals[-1],
            4 * gs.log(10) / nonzero_vals[0],
            n_domain,
        ),
        device,
    )


class WksDefaultDomain:
    """Compute WKS domain.

    Parameters
    ----------
    shape : Shape.
        Shape with basis.
    n_domain : int
        Number of energy points to use.
    n_overlap : int
        Controls Gaussian overlap. Ignored if ``sigma`` is not None.
    n_trans : int
        Number of standard deviations to translate energy bound by.
    """

    def __init__(self, n_domain, sigma=None, n_overlap=7, n_trans=2):
        self.n_domain = n_domain
        self.sigma = sigma
        self.n_overlap = n_overlap
        self.n_trans = n_trans

    def __call__(self, shape):
        """Compute WKS domain.

        Parameters
        ----------
        shape : Shape.
            Shape with basis.

        Returns
        -------
        domain : array-like, shape=[n_domain]
        sigma : float
            Standard deviation.
        """
        nonzero_vals = shape.basis.nonzero_vals
        device = getattr(nonzero_vals, "device", None)

        e_min, e_max = gs.log(nonzero_vals[0]), gs.log(nonzero_vals[-1])

        sigma = (
            self.n_overlap * (e_max - e_min) / self.n_domain
            if self.sigma is None
            else self.sigma
        )

        e_min += self.n_trans * sigma
        e_max -= self.n_trans * sigma

        energy = xgs.to_device(gs.linspace(e_min, e_max, self.n_domain), device)

        return energy, sigma

class HeatKernelSignature( WhichRegistryMixins, SpectralDescriptor):
    """Heat kernel signature.

    Parameters
    ----------
    scale : bool
        Whether to scale weights to sum to one.
    n_domain : int
        Number of domain points. Ignored if ``domain`` is not None.
    domain : callable or array-like, shape=[n_domain]
        Method to compute domain points (``f(shape)``) or
        domain points.
    """
        
    _Registry = HeatKernelSignatureRegistry

    def __init__(self, scale=True, n_domain=3, domain=None):
        super().__init__(domain or (lambda shape: hks_default_domain(shape, n_domain=n_domain)), scale=scale)

    def __call__(self, shape):
        """Compute descriptor.

        Parameters
        ----------
        shape : Shape.
            Shape with basis.

        Returns
        -------
        descr : array-like, shape=[n_domain, n_vertices]
            Descriptor.
        """
        domain = self.domain(shape) if callable(self.domain) else self.domain
        evals = shape.basis.vals
        vecs = shape.basis.vecs

        coefs = gs.exp(-la.scalarvecmul(domain, evals))
        coefs = self._scale(coefs)

        return self._compute_descriptor(coefs, vecs)

class WaveKernelSignature(WhichRegistryMixins, SpectralDescriptor):
    """Wave kernel signature."""

    _Registry = WaveKernelSignatureRegistry

    def __init__(self, scale=True, sigma=None, n_domain=3, domain=None):
        domain = domain or WksDefaultDomain(n_domain=n_domain, sigma=sigma)
        super().__init__(domain=domain, scale=scale)
        self.sigma = sigma

    def __call__(self, shape):
        """Compute descriptor.

        Parameters
        ----------
        shape : Shape.
            Shape with basis.

        Returns
        -------
        descr : array-like, shape=[n_domain, n_vertices]
            Descriptor.
        """
        if callable(self.domain):
            domain, sigma = self.domain(shape)
        else:
            domain, sigma = self.domain, self.sigma

        evals = shape.basis.nonzero_vals
        vecs = shape.basis.nonzero_vecs

        exp_arg = -xgs.square(gs.log(evals) - domain[:, None]) / (2 * xgs.square(sigma))
        coefs = gs.exp(exp_arg)
        coefs = self._scale(coefs)

        return self._compute_descriptor(coefs, vecs)

class LandmarkHeatKernelSignature(HeatKernelSignature):
    """Landmark-based Heat Kernel Signature.

    Parameters
    ----------
    scale : bool
        Whether to scale weights to sum to one.
    n_domain : int
        Number of domain points. Ignored if ``domain`` is not None.
    domain : callable or array-like, shape=[n_domain]
        Method to compute domain points (``f(shape)``) or
        domain points.
    """

    _Registry = LandmarkHeatKernelSignatureRegistry

    def __call__(self, shape):
        """Compute landmark-based HKS descriptor.

        Parameters
        ----------
        shape : Shape.
            Shape with basis and landmark_indices.

        Returns
        -------
        descr : array-like, shape=[n_landmarks * n_domain, n_vertices]
            Landmark-based HKS descriptor.
        """
        if not hasattr(shape, "landmark_indices") or shape.landmark_indices is None:
            raise AttributeError("Shape must have 'landmark_indices' set.")

        domain = self.domain(shape) if callable(self.domain) else self.domain
        evals = shape.basis.vals
        vecs = shape.basis.vecs
        landmarks = shape.landmark_indices

        # coefs: (n_domain, n_eigen)
        coefs = gs.exp(-gs.outer(domain, evals))
        return self._compute_landmark_descriptor(coefs, vecs, landmarks)

class LandmarkWaveKernelSignature(WaveKernelSignature):
    """Landmark-based Wave Kernel Signature.

    Parameters
    ----------
    scale : bool
        Whether to scale weights to sum to one.
    sigma : float
        Standard deviation for the Gaussian.
    n_domain : int
        Number of domain points. Ignored if ``domain`` is not None.
    domain : callable or array-like, shape=[n_domain]
        Method to compute domain points (``f(shape)``) or
        domain points.
    """

    _Registry = LandmarkWaveKernelSignatureRegistry

    def __call__(self, shape):
        """Compute landmark-based WKS descriptor.

        Parameters
        ----------
        shape : Shape.
            Shape with basis and landmark_indices.

        Returns
        -------
        descr : array-like, shape=[n_landmarks * n_domain, n_vertices]
            Landmark-based WKS descriptor.
        """
        if not hasattr(shape, "landmark_indices") or shape.landmark_indices is None:
            raise AttributeError("Shape must have 'landmark_indices' set.")

        if callable(self.domain):
            domain, sigma = self.domain(shape)
        else:
            domain, sigma = self.domain, self.sigma


        evals = shape.basis.nonzero_vals
        vecs = shape.basis.nonzero_vecs
        landmarks = shape.landmark_indices

        exp_arg = -xgs.square(gs.log(gs.abs(evals)) - domain[:, None]) / (2 * sigma**2)
        coefs = gs.exp(exp_arg)

        return self._compute_landmark_descriptor(coefs, vecs, landmarks)
