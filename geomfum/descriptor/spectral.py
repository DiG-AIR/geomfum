"""Spectral descriptors."""

import abc

import gsops.backend as gs

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
    """Compute HKS default domain. The domain is a set of sampled time points.

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
    return gs.to_device(
        gs.geomspace(
            4 * gs.log(10) / nonzero_vals[-1],
            4 * gs.log(10) / nonzero_vals[0],
            n_domain,
        ),
        device,
    ), None


class WksDefaultDomain:
    """Default domain generator for Wave Kernel Signature using logarithmic energy sampling.

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

        energy = gs.to_device(gs.linspace(e_min, e_max, self.n_domain), device)

        return energy, sigma


class SpectralFilter(abc.ABC):
    """Abstract base class for computing spectral filter coefficients from eigenvalues."""

    @abc.abstractmethod
    def __call__(self, vals, domain, sigma):
        """
        Compute filter coefficients for the given eigenvalues and domain.

        Parameters
        ----------
        vals : array-like, shape=[n_eigen]
            Eigenvalues.
        domain : array-like, shape=[n_domain]
            Domain points (e.g., time for HKS, energy for WKS).
        sigma : float or None
            Optional parameter for the filter (e.g., standard deviation for WKS).

        Returns
        -------
        coefs : array-like, shape=[n_domain, n_eigen]
            Filter coefficients.
        """


class HeatKernelFilter(SpectralFilter):
    """Heat kernel filter computing exp(-t * λ) coefficients for HKS."""

    def __call__(self, vals, domain, sigma):
        """
        Compute heat kernel filter coefficients.

        Parameters
        ----------
        vals : array-like, shape=[n_eigen]
            Eigenvalues.
        domain : array-like, shape=[n_domain]
            Time points.
        sigma : float or None
            Unused for heat kernel filter.

        Returns
        -------
        coefs : array-like, shape=[n_domain, n_eigen]
            Filter coefficients.
        """
        exp_arg = -la.scalarvecmul(domain, vals)
        return gs.exp(exp_arg)


class WaveKernelFilter(SpectralFilter):
    """Wave kernel filter using Gaussian weighting in log-eigenvalue space for WKS."""

    def __call__(self, vals, domain, sigma):
        """
        Compute wave kernel filter coefficients.

        Parameters
        ----------
        vals : array-like, shape=[n_eigen]
            Eigenvalues.
        domain : array-like, shape=[n_domain]
            Energy points (log-space).
        sigma : float
            Standard deviation for the Gaussian.

        Returns
        -------
        coefs : array-like, shape=[n_domain, n_eigen]
            Filter coefficients.
        """
        nonzero_vals = vals[gs.sum(gs.isclose(vals, 0.0, atol=1e-3)) :]
        zeros = gs.to_device(
            gs.zeros((domain.shape[0], vals.shape[0] - nonzero_vals.shape[0])),
            device=getattr(nonzero_vals, "device", None),
        )
        exp_arg = -gs.square(gs.log(nonzero_vals) - domain[:, None]) / (
            2 * gs.square(sigma)
        )
        coefs = gs.exp(exp_arg)

        if zeros.shape[1] > 0:
            coefs = gs.concatenate([zeros, coefs], axis=1)

        return coefs


class HeatKernelSignature(WhichRegistryMixins, SpectralDescriptor):
    """Heat Kernel Signature descriptor using heat diffusion over time.

    Parameters
    ----------
    scale : bool
        Whether to scale weights to sum to one.
    n_domain : int
        Number of domain points. Ignored if ``domain`` is not None.
    domain : callable or array-like, shape=[n_domain], optional
        Method to compute time domain points (``f(shape)``) or time domain points.
    k : int, optional
        Number of eigenfunctions to use. If None, all eigenfunctions are used.
    """

    _Registry = HeatKernelSignatureRegistry

    def __init__(self, scale=True, n_domain=3, domain=None, k=None):
        super().__init__(
            spectral_filter=HeatKernelFilter(),
            domain=domain
            or (lambda shape: hks_default_domain(shape, n_domain=n_domain)),
            scale=scale,
            sigma=1,
            landmarks=False,
            k=k,
        )


class WaveKernelSignature(WhichRegistryMixins, SpectralDescriptor):
    """Wave Kernel Signature descriptor using quantum mechanical wave propagation.

    Parameters
    ----------
    scale : bool
        Whether to scale weights to sum to one.
    sigma : float
        Standard deviation for the Gaussian.
    n_domain : int
        Number of domain points. Ignored if ``domain`` is not None.
    domain : callable or array-like, shape=[n_domain], optional
        Method to compute energy domain points (``f(shape)``) or energy domain points.
    k : int, optional
        Number of eigenfunctions to use. If None, all eigenfunctions are used.
    """

    _Registry = WaveKernelSignatureRegistry

    def __init__(self, scale=True, sigma=None, n_domain=3, domain=None, k=None):
        domain = domain or WksDefaultDomain(n_domain=n_domain, sigma=sigma)
        super().__init__(
            spectral_filter=WaveKernelFilter(),
            domain=domain,
            scale=scale,
            sigma=sigma,
            landmarks=False,
            k=k,
        )


class LandmarkHeatKernelSignature(WhichRegistryMixins, SpectralDescriptor):
    """Heat Kernel Signature computed from landmark points.

    Parameters
    ----------
    scale : bool
        Whether to scale weights to sum to one.
    n_domain : int
        Number of domain points. Ignored if ``domain`` is not None.
    domain : callable or array-like, shape=[n_domain], optional
        Method to compute time domain points (``f(shape)``) or time domain points.
    k : int, optional
        Number of eigenfunctions to use.
    """

    _Registry = LandmarkHeatKernelSignatureRegistry

    def __init__(self, scale=True, n_domain=3, domain=None, k=None):
        super().__init__(
            spectral_filter=HeatKernelFilter(),
            domain=domain
            or (lambda shape: hks_default_domain(shape, n_domain=n_domain)),
            scale=scale,
            sigma=1,
            landmarks=True,
            k=k,
        )


class LandmarkWaveKernelSignature(WhichRegistryMixins, SpectralDescriptor):
    """Wave Kernel Signature computed from landmark points.

    Parameters
    ----------
    scale : bool
        Whether to scale weights to sum to one.
    sigma : float
        Standard deviation for the Gaussian.
    n_domain : int
        Number of domain points. Ignored if ``domain`` is not None.
    domain : callable or array-like, shape=[n_domain], optional
        Method to compute energy domain points (``f(shape)``) or energy domain points.
    k : int, optional
        Number of eigenfunctions to use.
    """

    _Registry = LandmarkWaveKernelSignatureRegistry

    def __init__(self, scale=True, sigma=None, n_domain=3, domain=None, k=None):
        super().__init__(
            spectral_filter=WaveKernelFilter(),
            domain=domain or WksDefaultDomain(n_domain=n_domain, sigma=sigma),
            scale=scale,
            sigma=sigma,
            k=k,
            landmarks=True,
        )
