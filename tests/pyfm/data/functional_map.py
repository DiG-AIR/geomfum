import random

import numpy as np
from geomstats.test.data import TestData

from tests.utils import ShapePair


class WeightedFactorCmpTestData(TestData):
    trials = 1

    _indices = ["cat-00", "lion-00"]

    shape_pair = ShapePair(*_indices, target_reduction=0.95)

    def factor_value_random_test_data(self):
        return self.generate_tests([dict()])

    def factor_gradient_random_test_data(self):
        return self.generate_tests([dict()])

    def generate_random_descriptors(self, num_descr=None):
        shape_a, shape_b = self.shape_pair.get()

        num_descr = num_descr or random.randint(1, 6)
        descr_a = np.random.uniform(size=(num_descr, shape_a.n_vertices))
        descr_b = np.random.uniform(size=(num_descr, shape_b.n_vertices))

        return descr_a, descr_b
