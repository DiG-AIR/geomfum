"""Factors to build functional map objective function."""

import abc

import gsops.backend as gs

import geomfum.linalg as la
from geomfum.numerics.optimization import ScipyMinimize


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
        """Compute the commutativity constrain as a constrai on the spectrum coefficient.

        Parameters
        ----------
        basis_a: Basis
            Basis of source shape
        basis_b: Basis
            Basis of target shape

        Returns
        -------
        constraint: LBCommutativityEnforcings
            class for the LBCommutativityEnforcings constraint
        """
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
        Compute the OperatorCommutativityEnforcing constrain from the operator induced by descriptors.

        Parameters
        ----------
        basis_a : Basis
            Basis of the source shape.
        descr_a : array-like, shape=[..., n_vertices]
            descriptor for the source shape.
        basis_b : Basis
            Basis of the target shape.
        descr_b : array-like, shape=[..., n_vertices]
            descriptor for the target shape.
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
        Compute the OperatorCommutativityEnforcing constrain from the operator induced by gradient Operators.

        Parameters
        ----------
        basis_a : Basis
            Basis of the source shape.
        descr_a : array-like, shape=[..., n_vertices]
            descriptor for the source shape.
        basis_b : Basis
            Basis of the target shape.
        descr_b : array-like, shape=[..., n_vertices]
            descriptor for the target shape.
        reversing_a : bool
            whether to return operators associated to orientation inversion instead
                of orientation preservation (return the opposite of the second operator) for source shape
        reversing_b : bool
            whether to return operators associated to orientation inversion instead
                of orientation preservation (return the opposite of the second operator) for target shape
        normalize : bool
            whether to normalize the gradient on each face.
        weight : float
            Weight of the factor.


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


class FactorBuilder(abc.ABC):
    """Abstract base class for factor builders."""

    def __init__(self, weight=1.0):
        self.weight = weight

    @abc.abstractmethod
    def build(self, basis_a, basis_b, descr_a, descr_b):
        """Build factor from shape data.

        Parameters
        ----------
        basis_a : LaplaceEigenBasis
            Basis of source shape.
        basis_b : LaplaceEigenBasis
            Basis of target shape.
        descr_a : array-like, shape=[..., n_vertices_a]
            Descriptors on source shape.
        descr_b : array-like, shape=[..., n_vertices_b]
            Descriptors on target shape.

        Returns
        -------
        factor : WeightedFactor
        """


class SDPFactorBuilder(FactorBuilder):
    """Builder for SpectralDescriptorPreservation factor.

    Parameters
    ----------
    weight : float
        Weight of the factor.
    """

    def __init__(self, weight=1.0):
        super(SDPFactorBuilder, self).__init__(weight=weight)

    def build(self, basis_a, basis_b, descr_a, descr_b):
        """Build SpectralDescriptorPreservation from shape data.

        Parameters
        ----------
        basis_a : LaplaceEigenBasis
            Basis of source shape.
        basis_b : LaplaceEigenBasis
            Basis of target shape.
        descr_a : array-like, shape=[..., n_vertices_a]
            Descriptors on source shape.
        descr_b : array-like, shape=[..., n_vertices_b]
            Descriptors on target shape.

        Returns
        -------
        factor : SpectralDescriptorPreservation
        """
        return SpectralDescriptorPreservation(
            basis_a.project(descr_a),
            basis_b.project(descr_b),
            weight=self.weight,
        )


class LBCFactorBuilder(FactorBuilder):
    """Builder for LBCommutativityEnforcing factor.

    Parameters
    ----------
    weight : float
        Weight of the factor.
    """

    def __init__(self, weight=1e-2):
        super(LBCFactorBuilder, self).__init__(weight=weight)

    def build(self, basis_a, basis_b, descr_a, descr_b):
        """Build LBCommutativityEnforcing from bases.

        Parameters
        ----------
        basis_a : LaplaceEigenBasis
            Basis of source shape.
        basis_b : LaplaceEigenBasis
            Basis of target shape.
        descr_a : array-like
            Descriptors on source shape (ignored, for uniform interface).
        descr_b : array-like
            Descriptors on target shape (ignored, for uniform interface).

        Returns
        -------
        factor : LBCommutativityEnforcing
        """
        return LBCommutativityEnforcing.from_bases(basis_a, basis_b, weight=self.weight)


class MultFactorBuilder(FactorBuilder):
    """Builder for multiplication OperatorCommutativityEnforcing factor.

    Parameters
    ----------
    weight : float
        Weight of the factor.
    """

    def __init__(self, weight=1e-1):
        super(MultFactorBuilder, self).__init__(weight=weight)

    def build(self, basis_a, basis_b, descr_a, descr_b):
        """Build OperatorCommutativityEnforcing from multiplication operators.

        Parameters
        ----------
        basis_a : LaplaceEigenBasis
            Basis of source shape.
        basis_b : LaplaceEigenBasis
            Basis of target shape.
        descr_a : array-like, shape=[..., n_vertices_a]
            Descriptors on source shape.
        descr_b : array-like, shape=[..., n_vertices_b]
            Descriptors on target shape.

        Returns
        -------
        factor : OperatorCommutativityEnforcing
        """
        return OperatorCommutativityEnforcing.from_multiplication(
            basis_a, descr_a, basis_b, descr_b, weight=self.weight
        )


# =============================================================================
# Functional Map Optimizer - Intermediate abstraction layer
# =============================================================================


class FunctionalMap:
    """Optimizer for functional maps.

    Takes factor_builders as configuration, then optimizes fmap given shapes/descriptors.
    This is an intermediate abstraction between FactorBuilders and the high-level Matcher.

    Parameters
    ----------
    factor_builders : list[FactorBuilder], optional
        List of factor builders. If None, uses default (SDP + LB + Mult).
    optimizer : ScipyMinimize, optional
        Optimizer to use. If None, uses L-BFGS-B.
    """

    def __init__(self, fmap_size=None, factor_builders=None, optimizer=None):
        self.fmap_size = fmap_size
        self.factor_builders = factor_builders or self._default_factor_builders()
        self.optimizer = optimizer or ScipyMinimize(method="L-BFGS-B")

    def _default_factor_builders(self):
        """Return default factor builders.

        Returns
        -------
        builders : list[FactorBuilder]
            Default list of factor builders.
        """
        return [
            SDPFactorBuilder(weight=1.0),
            LBCFactorBuilder(weight=1e-2),
            MultFactorBuilder(weight=1e-1),
        ]

    def __call__(self, basis_a, basis_b, descr_a, descr_b, x_0=None):
        """Optimize functional map.

        Parameters
        ----------
        basis_a : LaplaceEigenBasis
            Basis of source shape.
        basis_b : LaplaceEigenBasis
            Basis of target shape.
        descr_a : array-like
            Descriptors on source shape.
        descr_b : array-like
            Descriptors on target shape.

        Returns
        -------
        fmap_matrix : array-like, shape=[spectrum_size_b, spectrum_size_a]
            Optimized functional map matrix.
        """
        if self.fmap_size is not None:
            if basis_a.spectrum_size != self.fmap_size[1]:
                basis_a.use_k = self.fmap_size[1]
            if basis_b.spectrum_size != self.fmap_size[0]:
                basis_b.use_k = self.fmap_size[0]

        # Build factors from builders
        objective = self._build_factor_sum(
            self.factor_builders, basis_a, basis_b, descr_a, descr_b
        )

        # Optimize
        if x_0 is None:
            x_0 = gs.zeros((basis_b.spectrum_size, basis_a.spectrum_size))

        res = self.optimizer.minimize(objective, x_0, fun_jac=objective.gradient)

        return res.x.reshape(x_0.shape)

    def _build_factor_sum(self, builders, basis_a, basis_b, descr_a, descr_b):
        """Build FactorSum from a list of factor builders.

        Parameters
        ----------
        builders : list[FactorBuilder]
            List of factor builders.
        basis_a : LaplaceEigenBasis
            Basis of source shape.
        basis_b : LaplaceEigenBasis
            Basis of target shape.
        descr_a : array-like
            Descriptors on source shape.
        descr_b : array-like
            Descriptors on target shape.

        Returns
        -------
        factor_sum : FactorSum
            Combined objective function.
        """
        factors = [
            builder.build(basis_a, basis_b, descr_a, descr_b) for builder in builders
        ]
        return FactorSum(factors)
