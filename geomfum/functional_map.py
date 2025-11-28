"""Factors to build functional map objective function."""

import abc

import gsops.backend as gs

import geomfum.linalg as la


class WeightedFactor(abc.ABC):
    """Weighted factor.

    Parameters
    ----------
    weight : float
        Weight of the factor.
    """

    def __init__(self, weight):
        self.weight = weight

    @abc.abstractmethod
    def __call__(self, fmap_matrix):
        """Compute energy.

        Parameters
        ----------
        fmap_matrix : array-like, shape=[spectrum_size_b, spectrum_size_a]
            Functional map matrix.

        Returns
        -------
        weighted_energy : float
            Weighted energy associated with the factor.
        """

    @abc.abstractmethod
    def gradient(self, fmap_matrix):
        """Compute energy gradient wrt functional map matrix.

        Parameters
        ----------
        fmap_matrix : array-like, shape=[spectrum_size_b, spectrum_size_a]
            Functional map matrix.

        Returns
        -------
        energy_gradient : array-like, shape=[spectrum_size_b, spectrum_size_a]
            Weighted energy gradient wrt functional map matrix.
        """


class SpectralDescriptorPreservation(WeightedFactor):
    """Spectral descriptor energy preservation.

    Parameters
    ----------
    sdescr_a : array-like, shape=[..., spectrum_size_a]
        Spectral descriptors on first basis.
    sdescr_a : array-like, shape=[..., spectrum_size_b]
        Spectral descriptors on second basis.
    weight : float
        Weight of the factor.
    """

    def __init__(self, sdescr_a, sdescr_b, weight=1.0):
        super().__init__(weight)
        self.sdescr_a = sdescr_a
        self.sdescr_b = sdescr_b

    def __call__(self, fmap_matrix):
        """Compute energy.

        Parameters
        ----------
        fmap_matrix : array-like, shape=[spectrum_size_b, spectrum_size_a]
            Functional map matrix.

        Returns
        -------
        weighted_energy : float
            Weighted descriptor preservation squared norm.
        """
        out = 0.5 * gs.square(la.matvecmul(fmap_matrix, self.sdescr_a) - self.sdescr_b)
        if out.ndim > 0:
            out = out.sum()

        return self.weight * out

    def gradient(self, fmap_matrix):
        """Compute energy gradient wrt functional map matrix.

        Parameters
        ----------
        fmap_matrix : array-like, shape=[spectrum_size_b, spectrum_size_a]
            Functional map matrix.

        Returns
        -------
        energy_gradient : array-like, shape=[spectrum_size_b, spectrum_size_a]
            Weighted energy gradient wrt functional map matrix.
        """
        out = gs.outer(
            la.matvecmul(fmap_matrix, self.sdescr_a) - self.sdescr_b, self.sdescr_a
        )

        if out.ndim > 2:
            out = out.sum(axis=tuple(range(out.ndim - 2)))
        return self.weight * out


class LBCommutativityEnforcing(WeightedFactor):
    """Laplace-Beltrami commutativity constraint.

    Parameters
    ----------
    ev_sqdiff : array-like, shape=[spectrum_size_b, spectrum_size_a]
        (Normalized) matrix of squared eigenvalue differences.
    weight : float
        Weight of the factor.
    """

    def __init__(self, vals_sqdiff, weight=1.0):
        super().__init__(weight)
        self.vals_sqdiff = vals_sqdiff

    @staticmethod
    def from_bases(basis_a, basis_b, weight=1.0):
        vals_sqdiff = gs.square(basis_a.vals[None, :] - basis_b.vals[:, None])
        vals_sqdiff /= vals_sqdiff.sum()
        return LBCommutativityEnforcing(vals_sqdiff, weight=weight)

    def __call__(self, fmap_matrix):
        """Compute energy.

        Parameters
        ----------
        fmap_matrix : array-like, shape=[spectrum_size_b, spectrum_size_a]
            Functional map matrix.

        Returns
        -------
        weighted_energy : float
            Weighted LB commutativity squared norm.
        """
        return self.weight * 0.5 * (gs.square(fmap_matrix) * self.vals_sqdiff).sum()

    def gradient(self, fmap_matrix):
        """Compute energy gradient wrt functional map matrix.

        Parameters
        ----------
        fmap_matrix : array-like, shape=[spectrum_size_b, spectrum_size_a]
            Functional map matrix.

        Returns
        -------
        energy_gradient : array-like, shape=[spectrum_size_b, spectrum_size_a]
            Weighted energy gradient wrt functional map matrix.
        """
        return self.weight * fmap_matrix * self.vals_sqdiff


class OperatorCommutativityEnforcing(WeightedFactor):
    """Operator commutativity constraint.

    Parameters
    ----------
    oper_a : array-like, shape=[spectrum_size_a, spectrum_size_a]
        Operator on first basis.
    oper_b : array-like, shape=[spectrum_size_b, spectrum_size_b]
        Operator on second basis.
    weight : float
        Weight of the factor.
    """

    def __init__(self, oper_a, oper_b, weight=1.0):
        super().__init__(weight)
        self.oper_a = oper_a
        self.oper_b = oper_b

    def __new__(cls, oper_a, oper_b, weight=1.0):
        """Create new instance of the operator.

        Parameters
        ----------
        oper_a : array-like, shape=[..., spectrum_size_a, spectrum_size_a]
            Operator on first basis.
        oper_b : array-like, shape=[..., spectrum_size_b, spectrum_size_b]
            Operator on second basis.
        weight : float
            Weight of the factor.

        Returns
        -------
        factor : OperatorCommutativityEnforcing or FactorSum
            Weighted factor.
        """
        if oper_a.ndim > 2:
            factors = [
                OperatorCommutativityEnforcing(oper_a_, oper_b_)
                for oper_a_, oper_b_ in zip(oper_a, oper_b)
            ]
            return FactorSum(factors, weight=weight)

        return super().__new__(cls)

    @staticmethod
    def compute_multiplication_operator(basis, descr):
        """Compute the multiplication operators associated with the descriptors.

        Parameters
        ----------
        descr : array-like, shape=[..., n_vertices]

        Returns
        -------
        operators : array-like, shape=[..., spectrum_size, spectrum_size]
        """
        return basis.pinv @ la.rowwise_scaling(descr, basis.vecs)

    @staticmethod
    def compute_orientation_operator(shape, descr, reversing=False, normalize=False):
        """
        Compute orientation preserving or reversing operators associated to each descriptor.

        Parameters
        ----------
        reversing : bool
            whether to return operators associated to orientation inversion instead
                    of orientation preservation (return the opposite of the second operator)
        normalize : bool
            whether to normalize the gradient on each face. Might improve results
                    according to the authors

        Returns
        -------
        list_op : list
            (n_descr,) where term i contains (D1,D2) respectively of size (k1,k1) and
            (k2,k2) which represent operators supposed to commute.
        """
        # Precompute the inverse of the eigenvectors matrix
        pinv = shape.basis.pinv  # (k1,n)

        # Compute the gradient of each descriptor
        grads = shape.face_valued_gradient(descr)
        if normalize:
            grads = la.normalize(grads)

        # Compute the operators in reduced basis
        sign = -1 if reversing else 1.0

        orients = shape.face_orientation_operator(grads)
        if descr.ndim > 1:
            return gs.stack(
                [sign * pinv @ orient @ shape.basis.vecs for orient in orients]
            )

        return sign * pinv @ orients @ shape.basis.vecs

    @classmethod
    def from_multiplication(cls, basis_a, descr_a, basis_b, descr_b, weight=1.0):
        """

        Parameters
        ----------
        descr_a : array-like, shape=[..., n_vertices]
        descr_b : array-like, shape=[..., n_vertices]
        """
        oper_a = cls.compute_multiplication_operator(basis_a, descr_a)
        oper_b = cls.compute_multiplication_operator(basis_b, descr_b)
        return OperatorCommutativityEnforcing(oper_a, oper_b, weight=weight)

    @classmethod
    def from_orientation(
        cls,
        shape_a,
        descr_a,
        shape_b,
        descr_b,
        reversing_a=False,
        reversing_b=False,
        normalize=False,
        weight=1.0,
    ):
        """

        Parameters
        ----------
        descr_a : array-like, shape=[..., n_vertices]
        descr_b : array-like, shape=[..., n_vertices]
        """
        oper_a = cls.compute_orientation_operator(
            shape_a, descr_a, reversing=reversing_a, normalize=normalize
        )
        oper_b = cls.compute_orientation_operator(
            shape_b, descr_b, reversing=reversing_b, normalize=normalize
        )
        return OperatorCommutativityEnforcing(oper_a, oper_b, weight=weight)

    def __call__(self, fmap_matrix):
        """Compute energy.

        Parameters
        ----------
        fmap_matrix : array-like, shape=[spectrum_size_b, spectrum_size_a]
            Functional map matrix.

        Returns
        -------
        energy : float
            Weighted operator commutativity squared norm.
        """
        return (
            self.weight
            * 0.5
            * gs.square(fmap_matrix @ self.oper_a - self.oper_b @ fmap_matrix).sum()
        )

    def gradient(self, fmap_matrix):
        """Compute energy gradient wrt functional map matrix.

        Parameters
        ----------
        fmap_matrix : array-like, shape=[spectrum_size_b, spectrum_size_a]
            Functional map matrix.

        Returns
        -------
        energy_gradient : array-like, shape=[spectrum_size_b, spectrum_size_a]
            Weighted energy gradient wrt functional map matrix.
        """
        return self.weight * (
            self.oper_b.T @ (self.oper_b @ fmap_matrix - fmap_matrix @ self.oper_a)
            - (self.oper_b @ fmap_matrix - fmap_matrix @ self.oper_a) @ self.oper_a.T
        )


class FactorSum(WeightedFactor):
    """Factor sum.

    Parameters
    ----------
    factors : list[WeightedFactor]
        Factors.
    """

    def __init__(self, factors, weight=1.0):
        super().__init__(weight=weight)
        self.factors = factors

    def __call__(self, fmap_matrix):
        """Compute energy.

        Parameters
        ----------
        fmap_matrix : array-like, shape=[spectrum_size_b, spectrum_size_a]
            Functional map matrix.

        Returns
        -------
        weighted_energy : float
            Weighted energy associated with the factor.
        """
        return self.weight * gs.sum(
            gs.array([factor(fmap_matrix) for factor in self.factors])
        )

    def gradient(self, fmap_matrix):
        """Compute energy gradient wrt functional map matrix.

        Parameters
        ----------
        fmap_matrix : array-like, shape=[spectrum_size_b, spectrum_size_a]
            Functional map matrix.

        Returns
        -------
        energy_gradient : array-like, shape=[spectrum_size_b, spectrum_size_a]
            Weighted energy gradient wrt functional map matrix.
        """
        return self.weight * gs.sum(
            gs.stack([factor.gradient(fmap_matrix) for factor in self.factors]), axis=0
        )
