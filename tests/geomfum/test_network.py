"""Tests for the functional map network (Consistent ZoomOut / Limit Shape)."""

import numpy as np
import pytest
from scipy.spatial import ConvexHull

from geomfum.convert import FmFromP2pConverter
from geomfum.laplacian import LaplacianFinder, LaplacianSpectrumFinder
from geomfum.network import FunctionalMapNetwork, _euclidean_fps
from geomfum.shape import TriangleMesh

SPECTRUM_SIZE = 8
M_INIT = 4
N_POINTS = 30

EDGES = [(0, 1), (1, 2), (2, 0), (1, 0), (2, 1), (0, 2)]


def _random_sphere_mesh(seed, n_points=N_POINTS):
    rng = np.random.default_rng(seed)
    points = rng.normal(size=(n_points, 3))
    points /= np.linalg.norm(points, axis=1, keepdims=True)

    hull = ConvexHull(points)
    return TriangleMesh(points, hull.simplices)


@pytest.fixture(scope="module")
def shapes():
    laplacian_finder = LaplacianFinder.from_registry(mesh=True, which="geomfum")
    spectrum_finder = LaplacianSpectrumFinder(
        spectrum_size=SPECTRUM_SIZE, laplacian_finder=laplacian_finder
    )

    shapes_ = [_random_sphere_mesh(seed) for seed in range(3)]
    for shape in shapes_:
        shape.laplacian.find_spectrum(laplacian_spectrum_finder=spectrum_finder)

    return shapes_


@pytest.fixture
def network(shapes):
    converter = FmFromP2pConverter()

    fmn = FunctionalMapNetwork(shapes)
    for i, j in EDGES:
        p2p = np.arange(shapes[j].n_vertices)
        fmap = converter(
            p2p, shapes[i].basis.truncate(M_INIT), shapes[j].basis.truncate(M_INIT)
        )
        fmn.add_edge(i, j, fmap)

    fmn.M = M_INIT
    return fmn


def test_euclidean_fps():
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [10, 10, 10]], dtype=float)

    indices = _euclidean_fps(points, n_samples=3)

    assert indices.shape == (3,)
    assert len(set(indices.tolist())) == 3
    assert np.all((indices >= 0) & (indices < len(points)))


def test_compute_subsample(network):
    size = 10
    network.compute_subsample(size=size)

    assert network.subsample.shape == (network.n_shapes, size)
    for i, shape in enumerate(network.shapes):
        assert np.all((network.subsample[i] >= 0) & (network.subsample[i] < shape.n_vertices))


def test_set_isometries(network):
    network.set_isometries(M=M_INIT)

    for i, j in network.edges:
        if (j, i) not in network.conn:
            continue

        fm_ij = network.conn[(i, j)]
        fm_ji = network.conn[(j, i)]
        assert np.allclose(fm_ij, fm_ji.T) or np.allclose(fm_ji, fm_ij.T)


def test_extract_3_cycles_and_amat(network):
    network.extract_3_cycles()
    assert set(network.cycles) == {(0, 1, 2), (2, 1, 0)}

    network.compute_Amat()
    assert network.A.shape == (len(network.cycles), len(network.edges))
    assert len(network.A_sub) == len(network.edges)


def test_get_cycle_weight(network):
    network.extract_3_cycles()
    cost = network.get_cycle_weight(network.cycles[0], M=M_INIT)

    assert cost >= 0


@pytest.mark.parametrize("weight_type", ["adjacency", "icsm"])
def test_set_weights(network, weight_type):
    network.set_weights(weight_type=weight_type)

    assert network.weights.shape == (network.n_shapes, network.n_shapes)
    assert network.use_icsm == (weight_type == "icsm")


def test_compute_clb_and_cclb(network):
    network.set_weights(weight_type="adjacency")
    network.compute_W(M=M_INIT)
    assert network.W.shape == (network.n_shapes * M_INIT, network.n_shapes * M_INIT)

    network.compute_CLB()
    assert network.CLB.shape == (network.n_shapes, M_INIT, M_INIT)

    m = 3
    network.compute_CCLB(m)
    assert network.CCLB.shape == (network.n_shapes, M_INIT, m)
    assert network.cclb_eigenvalues.shape == (m,)


def test_get_csd_and_lb(network):
    network.set_weights(weight_type="adjacency")

    m = 3
    network.compute_CCLB(m)

    csd_area, csd_conformal = network.get_CSD(0)
    assert csd_area.shape == (m, m)
    assert csd_conformal.shape == (m, m)

    latent_basis = network.get_LB(0, complete=True)
    assert latent_basis.shape == (network.shapes[0].n_vertices, m)


def test_compute_p2p_and_maps(network):
    network.set_weights(weight_type="adjacency")
    network.compute_CCLB(3)

    network.compute_p2p(complete=True)
    for i, j in network.edges:
        assert network.p2p[(i, j)].shape == (network.shapes[j].n_vertices,)

    new_M = M_INIT + 1
    network.compute_maps(new_M, complete=True)
    for i, j in network.edges:
        assert network.conn[(i, j)].shape == (new_M, new_M)
    assert network.M == new_M


def test_zoomout_refine(network):
    network.zoomout_refine(
        nit=2,
        step=1,
        subsample=None,
        isometric=True,
        weight_type="adjacency",
        M_init=M_INIT,
    )

    assert network.M == M_INIT + 1
    for i, j in network.edges:
        assert network.conn[(i, j)].shape == (M_INIT + 1, M_INIT + 1)
