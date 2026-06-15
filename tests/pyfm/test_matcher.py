import pytest
from polpo.testing import DataBasedParametrizer

from geomfum.matcher import FunctionalMapMatcher, ZoomOutMatcher
from tests.cases.cmp import MatcherCmpCase

from .data.matcher import MatcherCmpTestData

FMAP_SIZE = 20
SPECTRUM_SIZE = 120


@pytest.fixture(
    scope="class",
    params=["functional_map", "zoomout"],
)
def matchers(request):
    """Set ``matcher_a`` (geomfum) and ``matcher_b`` (pyFM) for each type.

    ``matcher_b`` is built through the registry, which exercises the
    geomfum <-> wrap wiring (``from_registry(which="pyfm")``).
    """
    matcher_type = request.param

    request.cls.testing_data.shapes.set_spectrum_finder(spectrum_size=SPECTRUM_SIZE)

    if matcher_type == "functional_map":
        request.cls.matcher_a = FunctionalMapMatcher(fmap_size=FMAP_SIZE)
        request.cls.matcher_b = FunctionalMapMatcher.from_registry(
            which="pyfm",
            fmap_size=FMAP_SIZE,
            k_process=SPECTRUM_SIZE,
            subsample_step=5,
        )

    elif matcher_type == "zoomout":
        request.cls.matcher_a = ZoomOutMatcher(fmap_size=(FMAP_SIZE, FMAP_SIZE))
        request.cls.matcher_b = ZoomOutMatcher.from_registry(
            which="pyfm",
            fmap_size=FMAP_SIZE,
            nit=5,
            step=2,
            k_process=SPECTRUM_SIZE,
            subsample_step=5,
        )

    else:
        raise ValueError(f"Unknown matcher type: {matcher_type}")


@pytest.mark.usefixtures("data_check", "matchers")
class TestMatcherCmp(MatcherCmpCase, metaclass=DataBasedParametrizer):
    testing_data = MatcherCmpTestData()
