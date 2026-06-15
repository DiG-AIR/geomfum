import gsops.backend as gs
import numpy as np
import pytest
from geomstats.test.test_case import TestCase


class LaplacianFinderCmpCase(TestCase):
    """Laplacian finder comparison.

    Notes
    -----
    Needs: `finder_a`, `finder_b`
    """

    @property
    def shapes(self):
        return self.testing_data.shapes

    def test_matrices_cmp(self, shape_key, atol):
        shape, other_shape = self.shapes.get(shape_key)

        laplacian_matrix, mass_matrix = self.finder_a(shape)
        laplacian_matrix_, mass_matrix_ = self.finder_b(other_shape)

        self.assertAllClose(
            gs.sparse.to_dense(laplacian_matrix),
            gs.sparse.to_dense(laplacian_matrix_),
            atol=atol,
        )
        self.assertAllClose(
            gs.sparse.to_dense(mass_matrix),
            gs.sparse.to_dense(mass_matrix_),
            atol=atol,
        )


class LaplacianSpectrumFinderCmpCase(TestCase):
    """Laplacian finder comparison.

    Notes
    -----
    Needs: `finder_a`, `finder_b`
    """

    @property
    def shapes(self):
        return self.testing_data.shapes

    def test_eigenvals_cmp(self, shape_key, atol):
        shape, other_shape = self.shapes.get(shape_key)

        vals, _ = self.finder_a(shape, as_basis=False)
        vals_, _ = self.finder_b(other_shape, as_basis=False)

        self.assertAllClose(vals, vals_, atol=atol)


class SpectralDescriptorCmpCase(TestCase):
    """Spectral descriptor computation comparison.

    Notes
    -----
    Needs: `descriptor_a`, `descriptor_b`
    """

    @property
    def shapes(self):
        return self.testing_data.shapes

    def test_descriptor_cmp(self, shape_key, atol):
        shape, other_shape = self.shapes.get(shape_key)

        descr_a = self.descriptor_a(shape)
        descr_b = self.descriptor_b(other_shape)

        self.assertAllClose(descr_a, descr_b, atol=atol)


class MatcherCmpCase(TestCase):
    """Shape matcher comparison (geomfum implementation vs external one).

    Both matchers map a shape to a copy of itself, which any correct functional
    map matcher should recover as the (near-)identity correspondence. This gives
    a deterministic, implementation-agnostic check that (a) each matcher is
    correct and (b) the geomfum and external implementations agree.

    Notes
    -----
    Needs: `matcher_a` (geomfum), `matcher_b` (external), and a `shapes`
    collection built with ``return_duplicate=True`` (self-pairs).
    """

    # self-matching a shape must recover (near-)identity; a broken matcher gives
    # essentially random indices (accuracy ~0), so this threshold is generous.
    min_self_match_accuracy = 0.95
    min_agreement = 0.95

    @property
    def shapes(self):
        return self.testing_data.shapes

    def _self_match_p2p(self, matcher, shape_key):
        shape, other = self.shapes.get(shape_key)
        p2p21 = gs.to_numpy(matcher(shape, other).p2p21)
        return p2p21, np.arange(shape.n_vertices)

    def test_self_match_identity(self, shape_key):
        for matcher in (self.matcher_a, self.matcher_b):
            p2p21, identity = self._self_match_p2p(matcher, shape_key)
            accuracy = (p2p21 == identity).mean()
            self.assertTrue(accuracy >= self.min_self_match_accuracy)

    def test_matchers_agree(self, shape_key):
        p2p_a, _ = self._self_match_p2p(self.matcher_a, shape_key)
        p2p_b, _ = self._self_match_p2p(self.matcher_b, shape_key)
        agreement = (p2p_a == p2p_b).mean()
        self.assertTrue(agreement >= self.min_agreement)


class WeightedFactorCmpCase(TestCase):
    """Factor computation comparison.

    Notes
    -----
    Needs: `factor_a`, `factor_b`
    """

    def _random_fmap_matrix(self):
        return np.random.uniform(size=(self.spectrum_size_b, self.spectrum_size_a))

    def _test_factor_value(self, fmap_matrix, atol):
        # TODO: identify bug when `test_factor_value`
        value = self.factor_a(fmap_matrix)
        value_ = self.factor_b(fmap_matrix)
        self.assertAllClose(value, value_, atol=atol)

    @pytest.mark.random
    def test_factor_value_random(self, atol):
        return self._test_factor_value(self._random_fmap_matrix(), atol)

    def _test_factor_gradient(self, fmap_matrix, atol):
        gradient = self.factor_a.gradient(fmap_matrix)
        gradient_ = self.factor_b.gradient(fmap_matrix)

        self.assertAllClose(gradient, gradient_, atol=atol)

    @pytest.mark.random
    def test_factor_gradient_random(self, atol):
        return self._test_factor_gradient(self._random_fmap_matrix(), atol)
