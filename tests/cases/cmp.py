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
