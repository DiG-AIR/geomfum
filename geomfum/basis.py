"""Basis implementations."""

import abc

import gsops.backend as gs

import geomfum.linalg as la


class Basis(abc.ABC):
    """Basis."""


class EigenBasis(Basis):
    """Eigenbasis.

    Parameters
    ----------
    vals : array-like, shape=[full_spectrum_size]
        Eigenvalues.
    vecs : array-like, shape=[dim, full_spectrum_size]
        Eigenvectors.
    use_k : int
        Number of values to use on computations.
    """

    def __init__(self, vals, vecs, use_k=None):
        self.full_vals = vals
        self.full_vecs = vecs
        self.use_k = use_k

        # NB: assumes sorted
        self._n_zeros = gs.sum(gs.isclose(vals, 0.0))

    @property
    def vals(self):
        """Eigenvalues.

        Returns
        -------
        vals : array-like, shape=[spectrum_size]
            Eigenvalues.
        """
        return self.full_vals[: self.use_k]

    @property
    def vecs(self):
        """Eigenvectors.

        Returns
        -------
        vecs : array-like, shape=[dim, spectrum_size]
            Eigenvectors.
        """
        return self.full_vecs[:, : self.use_k]

    @property
    def nonzero_vals(self):
        """Nonzero eigenvalues.

        Returns
        -------
        vals : array-like, shape=[spectrum_size - n_zeros]
            Eigenvalues.
        """
        return self.vals[self._n_zeros :]

    @property
    def nonzero_vecs(self):
        """Eigenvectors corresponding to nonzero eigenvalues.

        Returns
        -------
        vecs : array-like, shape=[dim, spectrum_size - n_zeros]
            Eigenvectors.
        """
        return self.vecs[:, self._n_zeros :]

    @property
    def spectrum_size(self):
        """Spectrum size.

        Returns
        -------
        spectrum_size : int
            Spectrum size.
        """
        return len(self.vals)

    @property
    def full_spectrum_size(self):
        """Full spectrum size.

        Returns
        -------
        spectrum_size : int
            Spectrum size.
        """
        return len(self.full_vals)

    def truncate(self, spectrum_size):
        """Truncate basis.

        Parameters
        ----------
        spectrum_size : int
            Spectrum size.

        Returns
        -------
        basis : Eigenbasis
            Truncated eigenbasis.
        """
        if spectrum_size == self.spectrum_size:
            return self

        return EigenBasis(self.vals[:spectrum_size], self.vecs[:, :spectrum_size])


class LaplaceEigenBasis(EigenBasis):
    """Laplace eigenbasis.

    Parameters
    ----------
    shape : Shape
        Shape.
    vals : array-like, shape=[spectrum_size]
        Eigenvalues.
    vecs : array-like, shape=[dim, spectrum_size]
        Eigenvectors.
    use_k : int
        Number of values to use on computations.
    """

    def __init__(self, shape, vals, vecs, use_k=None):
        super().__init__(vals, vecs, use_k)
        self._shape = shape

        self._pinv = None

    @property
    def use_k(self):
        """Number of values to use on computations.

        Returns
        -------
        use_k : int
            Number of values to use on computations.
        """
        return self._use_k

    @use_k.setter
    def use_k(self, value):
        """Set number of values to use on computations.

        Parameters
        ----------
        use_k : int
            Number of values to use on computations.
        """
        self._pinv = None
        self._use_k = value

    @property
    def pinv(self):
        """Inverse of the eigenvectors matrix.

        Return
        ------
        pinv : array-like, shape=[spectrum_size, n_vertices]
            Inverse of the eigenvectors matrix.
        """
        if self._pinv is None:
            self._pinv = self.vecs.T @ self._shape.laplacian.mass_matrix
        return self._pinv

    def truncate(self, spectrum_size):
        """Truncate basis.

        Parameters
        ----------
        spectrum_size : int
            Spectrum size.

        Returns
        -------
        basis : LaplaceEigenBasis
            Truncated eigenbasis.
        """
        if spectrum_size == self.spectrum_size:
            return self

        return LaplaceEigenBasis(
            self._shape,
            self.full_vals[:spectrum_size],
            self.full_vecs[:, :spectrum_size],
        )

    def project(self, array):
        """Project on the eigenbasis.

        Parameters
        ----------
        array : array-like, shape=[..., n_vertices]
            Array to project.

        Returns
        -------
        projected_array : array-like, shape=[..., spectrum_size]
            Projected array.
        """
        return la.matvecmul(
            self.vecs.T,
            la.matvecmul(self._shape.laplacian.mass_matrix, array),
        )
