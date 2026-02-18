"""Test Laplacian-related machinery for meshes."""

import random

import pytest
from polpo.testing import DataBasedParametrizer

from geomfum.laplacian import LaplacianFinder, LaplacianSpectrumFinder
from tests.cases.cmp import (
    LaplacianFinderCmpCase,
    LaplacianSpectrumFinderCmpCase,
)

from .data.laplacian import LaplacianFinderCmpData, LaplacianSpectrumFinderCmpData


@pytest.fixture(
    scope="class",
    params=[
        ("geomfum", "pyfm"),
        ("geomfum", "igl"),
    ],
)
def laplacian_finders(request):
    which_a, which_b = request.param

    request.cls.finder_a = LaplacianFinder.from_registry(which=which_a)
    request.cls.finder_b = LaplacianFinder.from_registry(which=which_b)


@pytest.mark.redundant
@pytest.mark.usefixtures("data_check", "laplacian_finders")
class TestLaplacianFinderCmp(LaplacianFinderCmpCase, metaclass=DataBasedParametrizer):
    """Laplacian finder comparison.

    Notes
    -----
    Redundant with ``TestLaplacianSpectrumFinderCmp``.
    """

    testing_data = LaplacianFinderCmpData()


@pytest.fixture(
    scope="class",
    params=[
        ("geomfum", "pyfm"),
        ("pyfm", "igl"),
        ("pyfm", "robust"),
        ("igl", "robust"),
    ],
)
def spectrum_finders(request):
    which_a, which_b = request.param

    spectrum_size = random.randint(2, 5)
    request.cls.finder_a = LaplacianSpectrumFinder(
        spectrum_size=spectrum_size,
        laplacian_finder=LaplacianFinder.from_registry(which=which_a),
    )
    request.cls.finder_b = LaplacianSpectrumFinder(
        spectrum_size=spectrum_size,
        laplacian_finder=LaplacianFinder.from_registry(which=which_b),
    )


@pytest.mark.usefixtures("data_check", "spectrum_finders")
class TestLaplacianSpectrumFinderCmp(
    LaplacianSpectrumFinderCmpCase, metaclass=DataBasedParametrizer
):
    """Laplacian spectrum finder comparison."""

    testing_data = LaplacianSpectrumFinderCmpData()
