"""Smoke tests for device management (Shape.to / .device protocol)."""

import gsops.backend as gs
import pytest
import torch

from tests.utils import get_meshes_from_indices

CUDA_AVAILABLE = torch.cuda.is_available()
# gs.to_device only moves data in pytorch backend mode; numpy mode is a no-op.
GSOPS_PYTORCH = "pytorch" in gs.__name__
CUDA_AND_PYTORCH = CUDA_AVAILABLE and GSOPS_PYTORCH


@pytest.fixture(scope="module")
def mesh():
    return get_meshes_from_indices(["cat-00"], target_reduction=0.95)[0]


@pytest.fixture(scope="module")
def torch_mesh():
    """Mesh with vertices/faces as torch tensors (needed for CUDA tests)."""
    shape = get_meshes_from_indices(["cat-00"], target_reduction=0.95)[0]
    # Convert to torch tensors so that gs.to_device can move to CUDA
    shape.vertices = torch.as_tensor(shape.vertices, dtype=torch.float64)
    shape.faces = torch.as_tensor(shape.faces, dtype=torch.int64)
    return shape


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _to_device_obj(d):
    """Normalize any device representation to torch.device."""
    if isinstance(d, torch.device):
        return d
    return torch.device(str(d))


def _device_of(tensor):
    """Return the device of a tensor as a torch.device."""
    raw = getattr(tensor, "device", "cpu")
    return _to_device_obj(raw)


def _assert_device(tensor, device, label=""):
    expected = torch.device(device)
    actual = _device_of(tensor)
    assert actual == expected, f"{label}: expected {expected}, got {actual}"


# ---------------------------------------------------------------------------
# .device property mirrors shape
# ---------------------------------------------------------------------------


class TestDeviceProperty:
    def test_shape_device_is_cpu_after_load(self, mesh):
        assert _to_device_obj(mesh.device) == torch.device("cpu")

    def test_laplacian_device_mirrors_shape(self, mesh):
        mesh.to("cpu")
        assert _to_device_obj(mesh.laplacian.device) == _to_device_obj(mesh.device)

    def test_gradient_device_mirrors_shape(self, mesh):
        mesh.to("cpu")
        assert _to_device_obj(mesh.gradient.device) == _to_device_obj(mesh.device)


# ---------------------------------------------------------------------------
# after find(), data lands on shape.device (CPU only)
# Scipy-backed matrices (stiffness, mass, gradient) always stay on CPU.
# ---------------------------------------------------------------------------


class TestLaplacianFindsOnDevice:
    def test_stiffness_on_cpu(self, mesh):
        mesh.to("cpu")
        mesh.laplacian.find(recompute=True)
        _assert_device(mesh.laplacian._stiffness_matrix, "cpu", "stiffness_matrix")
        _assert_device(mesh.laplacian._mass_matrix, "cpu", "mass_matrix")

    def test_spectrum_on_cpu(self, mesh):
        mesh.to("cpu")
        mesh.laplacian.find_spectrum(spectrum_size=10, recompute=True)
        _assert_device(mesh.laplacian._basis.full_vals, "cpu", "eigenvals")
        _assert_device(mesh.laplacian._basis.full_vecs, "cpu", "eigenvecs")

    @pytest.mark.skipif(not CUDA_AND_PYTORCH, reason="requires CUDA + gsops pytorch mode")
    def test_spectrum_on_cuda(self, torch_mesh):
        """Dense eigendata (vals, vecs) should move to CUDA; scipy sparse stays CPU."""
        torch_mesh.to("cpu")
        torch_mesh.laplacian.find_spectrum(spectrum_size=10, recompute=True)

        torch_mesh.to("cuda")
        # Dense tensors should be on CUDA
        _assert_device(torch_mesh.laplacian._basis.full_vals, "cuda", "eigenvals")
        _assert_device(torch_mesh.laplacian._basis.full_vecs, "cuda", "eigenvecs")
        # Scipy sparse stays CPU — this is by design
        _assert_device(
            torch_mesh.laplacian._stiffness_matrix, "cpu", "stiffness (stays cpu)"
        )
        torch_mesh.to("cpu")


# ---------------------------------------------------------------------------
# Shape.to() moves already-computed dense data
# ---------------------------------------------------------------------------


class TestShapeToMovesCaches:
    def test_to_cpu_returns_self(self, mesh):
        mesh.to("cpu")
        assert mesh.to("cpu") is mesh

    def test_vertices_on_cpu(self, mesh):
        mesh.to("cpu")
        _assert_device(mesh.vertices, "cpu", "vertices")

    def test_faces_on_cpu(self, mesh):
        mesh.to("cpu")
        _assert_device(mesh.faces, "cpu", "faces")

    def test_laplacian_matrices_on_cpu(self, mesh):
        mesh.to("cpu")
        mesh.laplacian.find()
        _assert_device(mesh.laplacian._stiffness_matrix, "cpu", "stiffness")

    def test_basis_on_cpu(self, mesh):
        mesh.to("cpu")
        mesh.laplacian.find_spectrum(spectrum_size=10, recompute=True)
        _assert_device(mesh.laplacian._basis.full_vecs, "cpu", "eigenvecs")

    @pytest.mark.skipif(not CUDA_AND_PYTORCH, reason="requires CUDA + gsops pytorch mode")
    def test_dense_round_trip_cpu_cuda_cpu(self, torch_mesh):
        """Vertices, faces, eigenvecs round-trip CPU→CUDA→CPU as torch tensors."""
        torch_mesh.to("cpu")
        torch_mesh.laplacian.find_spectrum(spectrum_size=10, recompute=True)

        torch_mesh.to("cuda")
        _assert_device(torch_mesh.vertices, "cuda", "vertices")
        _assert_device(torch_mesh.faces, "cuda", "faces")
        _assert_device(torch_mesh.laplacian._basis.full_vals, "cuda", "eigenvals")
        _assert_device(torch_mesh.laplacian._basis.full_vecs, "cuda", "eigenvecs")
        assert _to_device_obj(torch_mesh.device) == torch.device("cuda")

        torch_mesh.to("cpu")
        _assert_device(torch_mesh.vertices, "cpu", "vertices after back")
        _assert_device(torch_mesh.laplacian._basis.full_vals, "cpu", "eigenvals back")


# ---------------------------------------------------------------------------
# gradient
# ---------------------------------------------------------------------------


class TestGradientDevice:
    def test_gradient_matrix_on_cpu(self, mesh):
        mesh.to("cpu")
        _ = mesh.gradient.gradient_matrix  # trigger lazy compute
        _assert_device(mesh.gradient._gradient_matrix, "cpu", "gradient_matrix")

    @pytest.mark.skipif(not CUDA_AND_PYTORCH, reason="requires CUDA + gsops pytorch mode")
    def test_gradient_matrix_stays_cpu_on_cuda_shape(self, torch_mesh):
        """gradient_matrix is scipy sparse — stays CPU even when shape moves to CUDA."""
        torch_mesh.to("cpu")
        _ = torch_mesh.gradient.gradient_matrix  # compute on CPU
        torch_mesh.to("cuda")
        # scipy sparse can't go to CUDA — stays CPU by design
        _assert_device(torch_mesh.gradient._gradient_matrix, "cpu", "gradient (stays cpu)")
        torch_mesh.to("cpu")


# ---------------------------------------------------------------------------
# early-return guard: repeated .to(same_device) doesn't clear caches
# ---------------------------------------------------------------------------


class TestEarlyReturnGuard:
    def test_repeated_to_cpu_keeps_cache(self, mesh):
        mesh.to("cpu")
        mesh.laplacian.find(recompute=True)
        stiffness_id = id(mesh.laplacian._stiffness_matrix)

        mesh.to("cpu")  # should be a no-op (early-return guard)
        assert id(mesh.laplacian._stiffness_matrix) == stiffness_id, (
            "Repeated .to(cpu) should not invalidate cache"
        )
