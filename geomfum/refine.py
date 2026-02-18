"""Functional map refinement machinery."""

import abc
import logging

import gsops.backend as gs
import scipy

from geomfum.convert import (
    FmFromP2pBijectiveConverter,
    FmFromP2pConverter,
    NamFromP2pConverter,
    P2pFromFmConverter,
    P2pFromNamConverter,
    SinkhornP2pFromFmConverter,
)


class Refiner(abc.ABC):
    """Functional map refiner."""

    @abc.abstractmethod
    def __call__(self, fmap_matrix, basis_a, basis_b):
        """Apply refiner.

        Parameters
        ----------
        fmap_matrix : array-like, shape=[spectrum_size_b, spectrum_size_a]
            Functional map matrix.
        basis_a : Eigenbasis.
            Basis.
        basis_b: Eigenbasis.
            Basis.

        Returns
        -------
        fmap_matrix : array-like, shape=[spectrum_size_b, spectrum_size_a]
            Refined functional map matrix.
        """


class IdentityRefiner(Refiner):
    """A dummy refiner."""

    def __call__(self, fmap_matrix, basis_a=None, basis_b=None):
        """Apply refiner.

        Parameters
        ----------
        fmap_matrix : array-like, shape=[spectrum_size_b, spectrum_size_a]
            Functional map matrix.
        basis_a : Eigenbasis.
            Basis. Ignored.
        basis_b: Eigenbasis.
            Basis. Ignored.

        Returns
        -------
        fmap_matrix : array-like, shape=[spectrum_size_b, spectrum_size_a]
            Refined functional map matrix.
        """
        return fmap_matrix


class OrthogonalRefiner(Refiner):
    """Refinement using singular value decomposition.

    Parameters
    ----------
    flip_neg_det : bool
        Whether to flip negative determinant for square matrices.

    References
    ----------
    .. [OCSBG2012] Maks Ovsjanikov, Mirela Ben-Chen, Justin Solomon,
        Adrian Butscher, and Leonidas Guibas.
        “Functional Maps: A Flexible Representation of Maps between
        Shapes.” ACM Transactions on Graphics 31, no. 4 (2012): 30:1-30:11.
        https://doi.org/10.1145/2185520.2185526.
    """

    def __init__(self, flip_neg_det=True):
        self.flip_neg_det = flip_neg_det

    def __call__(self, fmap_matrix, basis_a=None, basis_b=None):
        """Apply refiner.

        Parameters
        ----------
        fmap_matrix : array-like, shape=[spectrum_size_b, spectrum_size_a]
            Functional map matrix.
        basis_a : Eigenbasis.
            Basis. Ignored.
        basis_b: Eigenbasis.
            Basis. Ignored.

        Returns
        -------
        fmap_matrix : array-like, shape=[spectrum_size_b, spectrum_size_a]
            Refined functional map matrix.
        """
        k2, k1 = fmap_matrix.shape
        U, _, VT = scipy.linalg.svd(fmap_matrix)

        if k1 != k2 or not self.flip_neg_det:
            return gs.asarray(U @ gs.to_numpy(gs.eye(k2, k1)) @ VT)

        opt_rot = gs.asarray(gs.matmul(U, VT))
        if gs.linalg.det(opt_rot) < 0.0:
            diag_sign = gs.diag(gs.ones(VT.shape[0]))
            diag_sign[-1, -1] = -1
            opt_rot = gs.matmul(U, gs.matmul(diag_sign, VT))

        return opt_rot


class ProperRefiner(Refiner):
    """Refinement projecting the functional map to the proper functional map space.

    Parameters
    ----------
    p2p_from_fm_converter : P2pFromFmConverter
        Pointwise map from functional map.
    fm_from_p2p_converter : FmFromP2pConverter
        Functional map from pointwise map.
    """

    def __init__(
        self,
        p2p_from_fm_converter=None,
        fm_from_p2p_converter=None,
    ):
        super().__init__()
        if p2p_from_fm_converter is None:
            p2p_from_fm_converter = P2pFromFmConverter()

        if fm_from_p2p_converter is None:
            fm_from_p2p_converter = FmFromP2pConverter()

        self.p2p_from_fm_converter = p2p_from_fm_converter
        self.fm_from_p2p_converter = fm_from_p2p_converter

    def __call__(self, fmap_matrix, basis_a, basis_b):
        """Apply refiner.

        Parameters
        ----------
        fmap_matrix : array-like, shape=[spectrum_size_b, spectrum_size_a]
            Functional map matrix.
        basis_a : Eigenbasis.
            Basis.
        basis_b: Eigenbasis.
            Basis.

        Returns
        -------
        fmap_matrix : array-like, shape=[spectrum_size_b, spectrum_size_a]
            Refined functional map matrix.
        """
        p2p_21 = self.p2p_from_fm_converter(fmap_matrix, basis_a, basis_b)
        return self.fm_from_p2p_converter(p2p_21, basis_a, basis_b)


class IterativeRefiner(Refiner):
    """Iterative refinement of functional map.

    At each iteration, it computes a pointwise map,
    converts it back to a functional map, and (optionally)
    furthers refines it.

    Parameters
    ----------
    nit : int
        Number of iterations.
    step : int or tuple[2, int]
        How much to increase each basis per iteration.
    atol : float
        Convergence tolerance.
        Ignored if step different than 1.
    p2p_from_fm_converter : P2pFromFmConverter
        Pointwise map from functional map.
    fm_from_p2p_converter : FmFromP2pConverter
        Functional map from pointwise map.
    iter_refiner : Refiner
        Refinement algorithm that runs within each iteration.
    """

    def __init__(
        self,
        nit=10,
        step=0,
        atol=None,
        p2p_from_fm_converter=None,
        fm_from_p2p_converter=None,
        iter_refiner=None,
    ):
        super().__init__()
        if p2p_from_fm_converter is None:
            p2p_from_fm_converter = P2pFromFmConverter()

        if fm_from_p2p_converter is None:
            fm_from_p2p_converter = FmFromP2pConverter()

        if iter_refiner is None:
            iter_refiner = IdentityRefiner()

        self.nit = nit
        self.step = step
        self.atol = atol
        self.p2p_from_fm_converter = p2p_from_fm_converter
        self.fm_from_p2p_converter = fm_from_p2p_converter
        self.iter_refiner = iter_refiner

        if self._step_a != self._step_b != 0 and atol is not None:
            raise ValueError("`atol` can't be used with step different than 0.")

    @property
    def step(self):
        """How much to increase each basis per iteration.

        Returns
        -------
        step : tuple[2, int]
            Step.
        """
        return self._step_a, self._step_b

    @step.setter
    def step(self, step):
        """Set step.

        Parameters
        ----------
        step : int or tuple[2, int]
            How much to increase each basis per iteration.
        """
        if isinstance(step, int):
            self._step_a = self._step_b = step
        else:
            self._step_a, self._step_b = step

    def iter(self, fmap_matrix, basis_a, basis_b):
        """Refiner iteration.

        Parameters
        ----------
        fmap_matrix : array-like, shape=[spectrum_size_b, spectrum_size_a]
            Functional map matrix.
        basis_a : Eigenbasis.
            Basis.
        basis_b: Eigenbasis.
            Basis.

        Returns
        -------
        fmap_matrix : array-like, shape=[spectrum_size_b + step_b, spectrum_size_a + step_a]
            Refined functional map matrix.
        """
        k2, k1 = fmap_matrix.shape
        new_k1, new_k2 = k1 + self._step_a, k2 + self._step_b

        p2p_21 = self.p2p_from_fm_converter(fmap_matrix, basis_a, basis_b)

        fmap_matrix = self.fm_from_p2p_converter(
            p2p_21, basis_a.truncate(new_k1), basis_b.truncate(new_k2)
        )
        return self.iter_refiner(fmap_matrix, basis_a, basis_b)

    def __call__(self, fmap_matrix, basis_a, basis_b):
        """Apply refiner.

        Parameters
        ----------
        fmap_matrix : array-like, shape=[spectrum_size_b, spectrum_size_a]
            Functional map matrix.
        basis_a : Eigenbasis.
            Basis.
        basis_b: Eigenbasis.
            Basis.

        Returns
        -------
        fmap_matrix : array-like, shape=[spectrum_size_b, spectrum_size_a]
            Refined functional map matrix.
        """
        k2, k1 = fmap_matrix.shape

        nit = self.nit
        if nit is None:
            nit = min(
                (k1 - basis_a.full_spectrum_size) // self._step_a,
                (k2 - basis_b.full_spectrum_size) // self._step_b,
            )
        else:
            msg = []
            if k1 + nit * self._step_a > basis_a.full_spectrum_size:
                msg.append("`basis_a`")
            if k2 + nit * self._step_b > basis_b.full_spectrum_size:
                msg.append("`basis_b`")

            if msg:
                raise ValueError(f"Not enough eigenvectors on {', '.join(msg)}.")

        for _ in range(nit):
            new_fmap_matrix = self.iter(fmap_matrix, basis_a, basis_b)

            if (
                self.atol is not None
                and gs.amax(gs.abs(new_fmap_matrix - fmap_matrix)) < self.atol
            ):
                break

            fmap_matrix = new_fmap_matrix

        else:
            if self.atol is not None:
                logging.warning(f"Maximum number of iterations reached: {nit}")

        return new_fmap_matrix


class IcpRefiner(IterativeRefiner):
    """Iterative refinement of functional map using SVD.

    Parameters
    ----------
    nit : int
        Number of iterations.
    atol : float
        Convergence tolerance.
    p2p_from_fm_converter : P2pFromFmConverter
        Pointwise map from functional map.
    fm_from_p2p_converter : FmFromP2pConverter
        Functional map from pointwise map.

    References
    ----------
    .. [OCSBG2012] Maks Ovsjanikov, Mirela Ben-Chen, Justin Solomon,
        Adrian Butscher, and Leonidas Guibas.
        “Functional Maps: A Flexible Representation of Maps between
        Shapes.” ACM Transactions on Graphics 31, no. 4 (2012): 30:1-30:11.
        https://doi.org/10.1145/2185520.2185526.
    """

    def __init__(
        self,
        nit=10,
        atol=1e-4,
        p2p_from_fm_converter=None,
        fm_from_p2p_converter=None,
    ):
        super().__init__(
            nit=nit,
            step=0,
            atol=atol,
            p2p_from_fm_converter=p2p_from_fm_converter,
            fm_from_p2p_converter=fm_from_p2p_converter,
            iter_refiner=OrthogonalRefiner(),
        )


class ZoomOut(IterativeRefiner):
    """Zoomout algorithm.

    Parameters
    ----------
    nit : int
        Number of iterations.
    step : int or tuple[2, int]
        How much to increase each basis per iteration.
    p2p_from_fm_converter : P2pFromFmConverter
        Pointwise map from functional map.
    fm_from_p2p_converter : FmFromP2pConverter
        Functional map from pointwise map.

    References
    ----------
    .. [MRRSWO2019] Simone Melzi, Jing Ren, Emanuele Rodolà, Abhishek Sharma,
        Peter Wonka, and Maks Ovsjanikov. “ZoomOut: Spectral Upsampling
        for Efficient Shape Correspondence.” arXiv, September 12, 2019.
        http://arxiv.org/abs/1904.07865
    """

    def __init__(
        self,
        nit=10,
        step=1,
        p2p_from_fm_converter=None,
        fm_from_p2p_converter=None,
    ):
        super().__init__(
            nit=nit,
            step=step,
            p2p_from_fm_converter=p2p_from_fm_converter,
            fm_from_p2p_converter=fm_from_p2p_converter,
            iter_refiner=None,
        )


class AdjointBijectiveZoomOut(ZoomOut):
    """Adjoint bijective zoomout algorithm.

    Parameters
    ----------
    nit : int
        Number of iterations.
    step : int or tuple[2, int]
        How much to increase each basis per iteration.

    References
    ----------
    :cite:`VM2024`
    """

    def __init__(
        self,
        nit=10,
        step=1,
    ):
        super().__init__(
            nit=nit,
            step=step,
            p2p_from_fm_converter=P2pFromFmConverter(adjoint=True, bijective=True),
            fm_from_p2p_converter=FmFromP2pBijectiveConverter(),
        )


class FastSinkhornFilters(ZoomOut):
    """Fast Sinkhorn filters.

    Parameters
    ----------
    nit : int
        Number of iterations.
    step : int or tuple[2, int]
        How much to increase each basis per iteration.
    neighbor_finder : SinkhornKNeighborsFinder
        Nearest neighbor finder.

    References
    ----------
    .. [PRMWO2021] Gautam Pai, Jing Ren, Simone Melzi, Peter Wonka, and Maks Ovsjanikov.
        "Fast Sinkhorn Filters: Using Matrix Scaling for Non-Rigid Shape Correspondence
        with Functional Maps." Proceedings of the IEEE/CVF Conference on Computer Vision
        and Pattern Recognition (CVPR), 2021, pp. 11956-11965.
        https://hal.science/hal-03184936/document
    """

    def __init__(
        self,
        nit=10,
        step=1,
        neighbor_finder=None,
    ):
        super().__init__(
            nit=nit,
            step=step,
            p2p_from_fm_converter=SinkhornP2pFromFmConverter(neighbor_finder),
            fm_from_p2p_converter=FmFromP2pConverter(),
        )


class NeuralZoomOut(ZoomOut):
    """Neural zoomout algorithm.

    Parameters
    ----------
    nit : int
        Number of iterations.
    step : int or tuple[2, int]
        How much to increase each basis per iteration.

    References
    ----------
    .. [VOM2025] Giulio Viganò, Maks Ovsjanikov, Simone Melzi.
        "NAM: Neural Adjoint Maps for refining shape correspondences".
    """

    def __init__(
        self,
        nit=10,
        step=1,
        device="cpu",
    ):
        super().__init__(
            nit=nit,
            step=step,
            p2p_from_fm_converter=P2pFromNamConverter(),
            fm_from_p2p_converter=NamFromP2pConverter(device=device),
        )
