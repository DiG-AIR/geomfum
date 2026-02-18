import pytest
from polpo.testing import DataBasedParametrizer
from pyFM.mesh import TriMesh

from geomfum.laplacian import LaplacianFinder
from tests.cases.cmp import LaplacianFinderCmpCase
from tests.geomfum.data.laplacian import LaplacianFinderCmpData


def _pyfm_finder(**kwargs):
    def finder(_cls, mesh):
        pyfm_mesh = TriMesh(mesh.vertices, mesh.faces)
        pyfm_mesh.process(k=0, **kwargs)

        return pyfm_mesh.W, pyfm_mesh.A

    return finder


@pytest.fixture(
    scope="class",
    params=[
        (dict(which="robust"), dict(robust=True)),
        (dict(which="pyfm"), dict(robust=False)),
    ],
)
def finders(request):
    kwargs_a, kwargs_b = request.param
    request.cls.finder_a = LaplacianFinder.from_registry(**kwargs_a)
    request.cls.finder_b = _pyfm_finder(**kwargs_b)


@pytest.mark.redundant
@pytest.mark.usefixtures("data_check", "finders")
class TestLaplacianFinderCmp(LaplacianFinderCmpCase, metaclass=DataBasedParametrizer):
    """Laplacian finder comparison.

    Notes
    -----
    Redundant with ``TestLaplacianSpectrumFinderCmp``.
    """

    testing_data = LaplacianFinderCmpData()
