"""Module for eigenvalue solver."""

import gsops.backend as gs
import scipy


class ScipyEigsh:
    """Sparse eigenvalue solver using SciPy's ARPACK wrapper.

    Parameters
    ----------
    spectrum_size : int, optional
        Number of eigenvalues and eigenvectors to compute (default: 6).
    sigma : float, optional
        Shift for shift-invert mode. If None, standard mode is used.
    which : str, optional
        Which eigenvalues to find: 'LM' (largest magnitude), 'SM' (smallest magnitude),
        'LA' (largest algebraic), 'SA' (smallest algebraic), etc. (default: 'LM').
    """

    def __init__(
        self,
        spectrum_size=6,
        sigma=None,
        which="LM",
    ):
        self.spectrum_size = spectrum_size
        self.sigma = sigma
        self.which = which

    def __call__(self, A, M=None):
        """Compute eigenvalues and eigenvectors.

        Parameters
        ----------
        A : array-like, sparse matrix
            The matrix for which to compute eigenvalues/eigenvectors.
        M : array-like, sparse matrix, optional
            Mass matrix for generalized eigenvalue problem A @ v = lambda * M @ v.

        Returns
        -------
        vals : array-like, shape=[spectrum_size]
            Eigenvalues.
        vecs : array-like, shape=[n, spectrum_size]
            Eigenvectors.
        """
        vals, vecs = scipy.sparse.linalg.eigsh(
            gs.sparse.to_scipy_csc(A),
            k=self.spectrum_size,
            M=gs.sparse.to_scipy_dia(M),
            sigma=self.sigma,
            which=self.which,
        )
        return gs.from_numpy(vals), gs.from_numpy(vecs)
