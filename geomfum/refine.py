"""Functional map refinement machinery."""

import abc
import logging

import numpy as np
import scipy

from geomfum.convert import (
    BijectiveP2pFromFmConverter,
    FmFromP2pBijectiveConverter,
    FmFromP2pConverter,
    P2pFromFmConverter,
    DiscreteOptimizationP2pFromFmConverter,
    SmoothP2pFromFmConverter,
    DirichletDisplacementFromP2pConverter,
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
            return U @ np.eye(k2, k1) @ VT

        opt_rot = np.matmul(U, VT)
        if np.linalg.det(opt_rot) < 0.0:
            diag_sign = np.diag(np.ones(VT.shape[0]))
            diag_sign[-1, -1] = -1
            opt_rot = np.matmul(U, np.matmul(diag_sign, VT))

        return opt_rot


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
                and np.amax(np.abs(new_fmap_matrix - fmap_matrix)) < self.atol
            ):
                break

            fmap_matrix = new_fmap_matrix

        else:
            if self.atol is not None:
                logging.warning(f"Maximum number of iterations reached: {nit}")

        return new_fmap_matrix


class BijectiveIterativeRefiner(Refiner):
    """Bijective Iterative refinement of functional map.

    At each iteration, it computes two pointwise map in both directions,
    converts it back to a pair of functional maps, and (optionally)
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
            p2p_from_fm_converter = BijectiveP2pFromFmConverter()

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

    def iter(self, fmap_matrix12, fmap_matrix21, basis_a, basis_b):
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
        k2, k1 = fmap_matrix12.shape
        new_k1, new_k2 = k1 + self._step_a, k2 + self._step_b

        p2p_21, p2p_12 = self.p2p_from_fm_converter(
            fmap_matrix12, fmap_matrix21, basis_a, basis_b
        )
        fmap_matrix12 = self.fm_from_p2p_converter(
            p2p_21, basis_a.truncate(new_k1), basis_b.truncate(new_k2)
        )
        fmap_matrix21 = self.fm_from_p2p_converter(
            p2p_12, basis_b.truncate(new_k2), basis_a.truncate(new_k1)
        )
        return self.iter_refiner(fmap_matrix12, basis_a, basis_b), self.iter_refiner(
            fmap_matrix21, basis_b, basis_a
        )

    def __call__(self, fmap_matrix12, fmap_matrix21, basis_a, basis_b):
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
        k2, k1 = fmap_matrix12.shape

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

        nit = self.nit

        for _ in range(nit):
            new_fmap_matrix12, new_fmap_matrix21 = self.iter(
                fmap_matrix12, fmap_matrix21, basis_a, basis_b
            )

            if (
                self.atol is not None
                and np.amax(np.abs(new_fmap_matrix12 - fmap_matrix12)) < self.atol
            ):
                break

            fmap_matrix12, fmap_matrix21 = new_fmap_matrix12, new_fmap_matrix21

        else:
            if self.atol is not None:
                logging.warning(f"Maximum number of iterations reached: {nit}")

        return new_fmap_matrix12, new_fmap_matrix21


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


class BijectiveZoomOut(BijectiveIterativeRefiner):
    """Bijective Zoomout algorithm.

    Parameters
    ----------
    nit : int
        Number of iterations.
    step : int or tuple[2, int]
        How much to increase each basis per iteration.
    p2p_from_fm_converter : BijectiveP2pFromFmConverter
        Pointwise map from functional map.
    fm_from_p2p_converter : FmFromP2pConverter
        Functional map from pointwise map.

    References
    ----------
    .. [RMOW2023] Jing Ren, Simone Melzi, Maks Ovsjanikov, Peter Wonka.
        "MapTree: Recovering Multiple Solutions in the Space of Maps."
        ACM Transactions on Graphics 42, no. 4 (2023): 1-15.
    """

    def __init__(
        self,
        nit=10,
        step=1,
    ):
        super().__init__(
            nit=nit,
            step=step,
            p2p_from_fm_converter=BijectiveP2pFromFmConverter(),
            fm_from_p2p_converter=FmFromP2pConverter(),
            iter_refiner=None,
        )


class DiscreteOptimization(BijectiveIterativeRefiner):
    """Discrete optimization refinement of functional map.


    Parameters
    ----------
    nit : int
        Number of iterations.
    step : int or tuple[2, int]
        How much to increase each basis per iteration.
    References
    ----------
    .. [RMWO2021] Jing Ren, Simone Melzi, Peter Wonka, Maks Ovsjanikov.
        “Discrete Optimization for Shape Matching.” Eurographics Symposium
        on Geometry Processing 2021, K. Crane and J. Digne (Guest Editors),
        Volume 40 (2021), Number 5.
    """

    def __init__(
        self, nit=10, step=1, energies=["ortho", "adjoint", "conformal", "descriptors"]
    ):
        super().__init__(
            nit=nit,
            step=step,
            p2p_from_fm_converter=DiscreteOptimizationP2pFromFmConverter(
                energies=energies
            ),
            fm_from_p2p_converter=FmFromP2pConverter(),
            iter_refiner=None,
        )

    def iter(
        self, fmap_matrix12, fmap_matrix21, basis_a, basis_b, descr_a=None, descr_b=None
    ):
        """Refiner iteration.

        Parameters
        ----------
        fmap_matrix : array-like, shape=[spectrum_size_b, spectrum_size_a]
            Functional map matrix.
        basis_a : Eigenbasis.
            Basis.
        basis_b: Eigenbasis.
            Basis.
        descr_a : array-like, shape=[n_vertices_a, n_descriptors]
            Descriptors for `basis_a`.
        descr_b : array-like, shape=[n_vertices_b, n_descriptors]
            Descriptors for `basis_b`.
        Returns
        -------
        fmap_matrix : array-like, shape=[spectrum_size_b + step_b, spectrum_size_a + step_a]
            Refined functional map matrix.
        """
        k2, k1 = fmap_matrix12.shape
        new_k1, new_k2 = k1 + self._step_a, k2 + self._step_b
        basis_a.use_k, basis_b.use_k = k1, k2
        p2p_21, p2p_12 = self.p2p_from_fm_converter(
            fmap_matrix12, fmap_matrix21, basis_a, basis_b, descr_a, descr_b
        )
        fmap_matrix12 = self.fm_from_p2p_converter(
            p2p_21, basis_a.truncate(new_k1), basis_b.truncate(new_k2)
        )
        fmap_matrix21 = self.fm_from_p2p_converter(
            p2p_12, basis_b.truncate(new_k2), basis_a.truncate(new_k1)
        )
        basis_a.use_k, basis_b.use_k = new_k1, new_k2
        return self.iter_refiner(fmap_matrix12, basis_a, basis_b), self.iter_refiner(
            fmap_matrix21, basis_b, basis_a
        )

    def __call__(
        self, fmap_matrix12, fmap_matrix21, basis_a, basis_b, descr_a=None, descr_b=None
    ):
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
        k2, k1 = fmap_matrix12.shape

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
            new_fmap_matrix12, new_fmap_matrix21 = self.iter(
                fmap_matrix12,
                fmap_matrix21,
                basis_a,
                basis_b,
                descr_a=descr_a,
                descr_b=descr_b,
            )

            if (
                self.atol is not None
                and np.amax(np.abs(new_fmap_matrix12 - fmap_matrix12)) < self.atol
            ):
                break

            fmap_matrix12, fmap_matrix21 = new_fmap_matrix12, new_fmap_matrix21

        else:
            if self.atol is not None:
                logging.warning(f"Maximum number of iterations reached: {nit}")

        return new_fmap_matrix12, new_fmap_matrix21


class SmoothOptimization(BijectiveIterativeRefiner):
    """SMooth Functional maps optimization and refinement.


    Parameters
    ----------
    nit : int
        Number of iterations.
    step : int or tuple[2, int]
        How much to increase each basis per iteration.
    References
    ----------
    .. [MRSO2022] R. Magnet, J. Ren, O. Sorkine-Hornung, and M. Ovsjanikov.
        "Smooth NonRigid Shape Matching via Effective Dirichlet Energy Optimization."
        In 2022 International Conference on 3D Vision (3DV).
    """

    def __init__(self, nit=10, step=1, w_coupling=1e3):
        super().__init__(
            nit=nit,
            step=step,
            p2p_from_fm_converter=SmoothP2pFromFmConverter(),
            fm_from_p2p_converter=FmFromP2pConverter(),
            iter_refiner=None,
        )
        self.w_coupling = w_coupling
        self.displ_from_p2p_converter = DirichletDisplacementFromP2pConverter(
            w_coupling=w_coupling
        )

    def iter(
        self,
        fmap_matrix12,
        fmap_matrix21,
        displ21,
        displ12,
        mesh_a,
        mesh_b,
        W_a,
        A_a,
        W_b,
        A_b,
    ):
        """Refiner iteration.

        Parameters
        ----------
        fmap_matrix12: array-like, shape=[spectrum_size_b, spectrum_size_a]
            Functional map matrix.
        fmap_matrix21: array-like, shape=[spectrum_size_a, spectrum_size_b]
            Functional map matrix.
        displ21: array-like, shape=[n_vertices_a, 3]
            Displacement matrix.
        displ12: array-like, shape=[n_vertices_b, 3]
            Displacement matrix.
        mesh_a : Mesh
            Mesh.
        mesh_b : Mesh
            Mesh.
        Returns
        -------
        fmap_matrix12 : array-like, shape=[spectrum_size_b + step_b, spectrum_size_a + step_a]
            Refined functional map matrix.
        fmap_matrix21 : array-like, shape=[spectrum_size_b + step_b, spectrum_size_a + step_a]
            Refined functional map matrix.
        displ21 : array-like, shape=[n_vertices_a, 3]
            Refined displacement matrix.
        displ12 : array-like, shape=[n_vertices_b, 3]
            Refined displacement matrix.
        """
        basis_a = mesh_a.basis
        basis_b = mesh_b.basis

        k2, k1 = fmap_matrix12.shape
        new_k1, new_k2 = k1 + self._step_a, k2 + self._step_b

        basis_a.use_k, basis_b.use_k = k1, k2
        p2p_21, p2p_12 = self.p2p_from_fm_converter(
            fmap_matrix12, fmap_matrix21, displ21, displ12, mesh_a, mesh_b
        )

        fmap_matrix12 = self.fm_from_p2p_converter(
            p2p_21, basis_a.truncate(new_k1), basis_b.truncate(new_k2)
        )
        fmap_matrix21 = self.fm_from_p2p_converter(
            p2p_12, basis_b.truncate(new_k2), basis_a.truncate(new_k1)
        )

        displ21 = self.displ_from_p2p_converter(p2p_21, mesh_a, mesh_b, W_b, A_b)
        displ12 = self.displ_from_p2p_converter(p2p_12, mesh_b, mesh_a, W_a, A_a)

        basis_a.use_k, basis_b.use_k = new_k1, new_k2

        return (
            self.iter_refiner(fmap_matrix12, basis_a, basis_b),
            self.iter_refiner(fmap_matrix21, basis_b, basis_a),
            displ21,
            displ12,
        )

    def __call__(
        self,
        fmap_matrix12,
        fmap_matrix21,
        displ21,
        displ12,
        mesh_a,
        mesh_b,
        W_a=None,
        A_a=None,
        W_b=None,
        A_b=None,
    ):
        """Apply refiner.

        Parameters
        ----------
        fmap_matrix12: array-like, shape=[spectrum_size_b, spectrum_size_a]
            Functional map matrix.
        fmap_matrix21: array-like, shape=[spectrum_size_a, spectrum_size_b]
            Functional map matrix.
        displ21: array-like, shape=[n_vertices_a, 3]
            Displacement matrix.
        displ12: array-like, shape=[n_vertices_b, 3]
            Displacement matrix.
        mesh_a : Mesh
            Mesh.
        mesh_b : Mesh
            Mesh.
        Returns
        -------
        fmap_matrix : array-like, shape=[spectrum_size_b, spectrum_size_a]
            Refined functional map matrix.
        """
        k2, k1 = fmap_matrix12.shape

        nit = self.nit
        if nit is None:
            nit = min(
                (k1 - mesh_a.basis.full_spectrum_size) // self._step_a,
                (k2 - mesh_b.basis.full_spectrum_size) // self._step_b,
            )
        else:
            msg = []
            if k1 + nit * self._step_a > mesh_a.basis.full_spectrum_size:
                msg.append("`basis_a`")
            if k2 + nit * self._step_b > mesh_b.basis.full_spectrum_size:
                msg.append("`basis_b`")

            if msg:
                raise ValueError(f"Not enough eigenvectors on {', '.join(msg)}.")

        nit = self.nit

        # compute laplacian utils
        if W_a is None or W_b is None or A_a is None or A_b is None:
            from geomfum.laplacian import LaplacianFinder

            laplacian_finder = LaplacianFinder.from_registry(which="robust")
            W_a, A_a = laplacian_finder(mesh_a)
            W_b, A_b = laplacian_finder(mesh_b)
        # first a pass of the refinement
        for _ in range(nit):
            new_fmap_matrix12, new_fmap_matrix21, new_displ21, new_displ12 = self.iter(
                fmap_matrix12,
                fmap_matrix21,
                displ21,
                displ12,
                mesh_a,
                mesh_b,
                W_a,
                A_a,
                W_b,
                A_b,
            )

            if (
                self.atol is not None
                and np.amax(np.abs(new_fmap_matrix12 - fmap_matrix12)) < self.atol
            ):
                break

            fmap_matrix12, fmap_matrix21 = new_fmap_matrix12, new_fmap_matrix21
            displ21, displ12 = new_displ21, new_displ12
        else:
            if self.atol is not None:
                logging.warning(f"Maximum number of iterations reached: {nit}")

        return new_fmap_matrix12, new_fmap_matrix21, new_displ21, new_displ12


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
    .. [VM2023] Giulio Viganò, Simone Melzi.
        Adjoint Bijective ZoomOut: Efficient upsampling for learned linearly-invariant
        embedding. 2023
        https://github.com/gviga/AB-ZoomOut
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
