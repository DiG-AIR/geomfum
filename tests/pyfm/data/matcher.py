from geomstats.test.data import TestData

from tests.utils import ShapeCollection


class MatcherCmpTestData(TestData):
    trials = 1

    _indices = ["cat-00"]
    # return_duplicate=True -> get(key) returns a self-pair (shape, shape)
    shapes = ShapeCollection(
        _indices, target_reduction=0.95, return_duplicate=True
    )

    def self_match_identity_test_data(self):
        data = [dict(shape_key=key) for key in self.shapes.keys()]
        return self.generate_tests(data)

    def matchers_agree_test_data(self):
        data = [dict(shape_key=key) for key in self.shapes.keys()]
        return self.generate_tests(data)
