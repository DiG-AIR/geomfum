from geomstats.test.data import TestData

from tests.utils import ShapeCollection


class LaplacianFinderCmpData(TestData):
    _indices = ["cat-00"]
    shapes = ShapeCollection(_indices, target_reduction=0.95, return_duplicate=True)

    tolerances = {"matrices_cmp": {"atol": 1e-3}}

    def matrices_cmp_test_data(self):
        data = [dict(shape_key=shape_key) for shape_key in self.shapes.keys()]
        return self.generate_tests(data)


class LaplacianSpectrumFinderCmpData(TestData):
    _indices = ["cat-00"]
    shapes = ShapeCollection(_indices, target_reduction=0.6, return_duplicate=True)

    tolerances = {"eigenvals_cmp": {"atol": 5e-1}}

    def eigenvals_cmp_test_data(self):
        data = [dict(shape_key=shape_key) for shape_key in self.shapes.keys()]
        return self.generate_tests(data)
