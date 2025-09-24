"""Shape dataset for PyTorch."""

import itertools
import os
import os.path as osp
import random

import geomstats.backend as gs
import numpy as np
import scipy
import torch
from torch.utils.data import Dataset

import geomfum.backend as xgs
from geomfum.metric import VertexEuclideanMetric
from geomfum.metric.mesh import ScipyGraphShortestPathMetric
from geomfum.shape import PointCloud, TriangleMesh

from ._utils import hash_arrays


def get_cached_shape_data(
    filepath, shape_type, spectral, k, cache_dir=None, overwrite_cache=False
):
    """
    Load shape with caching support for expensive operations.
    Similar to DiffusionNet's get_operators but for shape data.
    """
    if shape_type == "mesh":
        shape = TriangleMesh.from_file(filepath)
    else:
        shape = PointCloud.from_file(filepath)

    verts_np = gs.to_numpy(shape.vertices)
    faces_np = gs.to_numpy(shape.faces) if hasattr(shape, "faces") else None

    # Check for cached spectral data
    found = False
    if cache_dir and spectral:
        os.makedirs(cache_dir, exist_ok=True)

        # Create hash from vertices and faces
        hash_key_str = hash_arrays(verts_np, faces_np)

        i_cache = 0
        while True:
            cache_path = osp.join(cache_dir, f"shape_{hash_key_str}_{i_cache}.npz")

            try:
                npzfile = np.load(cache_path, allow_pickle=True)
                cache_verts = npzfile["vertices"]
                cache_faces = npzfile.get("faces", None)
                cache_k = npzfile["k_eig"].item()

                # Check if cache matches current shape
                if (not np.allclose(verts_np, cache_verts, rtol=1e-10)) or (
                    faces_np is not None and not np.array_equal(faces_np, cache_faces)
                ):
                    i_cache += 1
                    continue

                # Check if we need more eigenvectors
                if overwrite_cache or cache_k < k:
                    os.remove(cache_path)
                    break

                # Load cached spectral data
                evals = npzfile["eigenvalues"][:k]
                evecs = npzfile["eigenvectors"][:, :k]
                pinv = npzfile["pinv"][:k, :]

                def read_sparse_matrix(prefix):
                    data = npzfile[prefix + "_data"]
                    indices = npzfile[prefix + "_indices"]
                    indptr = npzfile[prefix + "_indptr"]
                    shape = npzfile[prefix + "_shape"]
                    return scipy.sparse.csc_matrix((data, indices, indptr), shape=shape)

                mass_matrix = read_sparse_matrix("mass")
                stiffness_matrix = read_sparse_matrix("stiffness")
                gradient_matrix = read_sparse_matrix("gradient")

                # Set cached spectral data on shape
                shape.basis.full_vals = gs.array(evals)
                shape.basis.full_vecs = gs.array(evecs)
                shape.basis.pinv = gs.array(pinv)

                # Convert scipy sparse matrices back to your format
                shape.laplacian._mass_matrix = xgs.from_scipy_sparse(mass_matrix)
                shape.laplacian._stiffness_matrix = xgs.from_scipy_sparse(
                    stiffness_matrix
                )
                # Instantiate a complex gradient matrix using gradient_x_matrix as real and gradient_y_matrix as imaginary part
                shape.gradient._gradient_matrix = xgs.from_scipy_sparse(gradient_matrix)

                found = True
                break

            except FileNotFoundError:
                break

    # Compute spectral data if not cached
    if spectral and not found:
        shape.laplacian.find_spectrum(spectrum_size=k, set_as_basis=True)
        shape.gradient.gradient_matrix
        if cache_dir:
            evals_np = gs.to_numpy(shape.basis.full_vals)
            evecs_np = gs.to_numpy(shape.basis.full_vecs)
            pinv_np = gs.to_numpy(shape.basis.pinv)

            mass_scipy = xgs.sparse.to_scipy_csc(shape.laplacian._mass_matrix)
            stiffness_scipy = xgs.sparse.to_scipy_csc(shape.laplacian._stiffness_matrix)

            mass_scipy = mass_scipy
            stiffness_scipy = stiffness_scipy

            gradient_scipy = xgs.sparse.to_scipy_csc(shape.gradient._gradient_matrix)
            np.savez(
                cache_path,
                vertices=verts_np,
                faces=faces_np.astype(np.int32) if faces_np is not None else None,
                k_eig=k,
                eigenvalues=evals_np,
                eigenvectors=evecs_np,
                pinv=pinv_np,
                # Mass matrix sparse components
                mass_data=mass_scipy.data,
                mass_indices=mass_scipy.indices,
                mass_indptr=mass_scipy.indptr,
                mass_shape=mass_scipy.shape,
                # Stiffness matrix sparse components
                stiffness_data=stiffness_scipy.data,
                stiffness_indices=stiffness_scipy.indices,
                stiffness_indptr=stiffness_scipy.indptr,
                stiffness_shape=stiffness_scipy.shape,
                # gradient matrix components
                gradient_data=gradient_scipy.data,
                gradient_indices=gradient_scipy.indices,
                gradient_indptr=gradient_scipy.indptr,
                gradient_shape=gradient_scipy.shape,
            )

    return shape


class ShapeDataset(Dataset):
    """General dataset for loading and preprocessing meshes or point clouds.

    Parameters
    ----------
    dataset_dir : str
        Path to the directory containing the dataset. We assume the dataset directory to have a subfolder shapes, for shapes, corr, for correspondences and dist, for cached distance matrices.
    shape_type : str
        Type of shape to load. Either 'mesh' or 'pointcloud'.
    spectral : bool
        Whether to compute the spectral features.
    distances : bool
        Whether to compute geodesic distance matrices. For computational reasons, these are not computed on the fly, but rather loaded from a precomputed .mat file.
    correspondences : bool
        Whether to load correspondences.
    k : int
        Number of eigenvectors to use for the spectral features.
    device : torch.device, optional
        Device to move the data to.
    """

    def __init__(
        self,
        dataset_dir,
        shape_type="mesh",
        spectral=False,
        distances=False,
        correspondences=True,
        k=200,
        device=None,
        cache_dir=None,
        overwrite_cache=False,
        preload_all=False,
    ):
        if shape_type not in ["mesh", "pointcloud"]:
            raise ValueError("shape_type must be either 'mesh' or 'pointcloud'")

        # basic attributes
        self.dataset_dir = dataset_dir
        self.shape_type = shape_type
        self.shape_dir = os.path.join(dataset_dir, "shapes")
        all_shape_files = sorted(
            [
                f
                for f in os.listdir(self.shape_dir)
                if f.lower().endswith((".off", ".ply", ".obj"))
            ]
        )
        self.shape_files = all_shape_files

        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.spectral = spectral
        self.k = k
        self.distances = distances
        self.correspondences = correspondences

        # cache options
        if cache_dir is None:
            self.cache_dir = os.path.join(dataset_dir, ".cache", "shapes")
        else:
            self.cache_dir = cache_dir
        self.overwrite_cache = overwrite_cache
        self.preload_all = preload_all
        if preload_all:
            # Original behavior
            self.shapes = {}
            self.corrs = {}
            self._preload_all_shapes()
        else:
            # Lazy loading with persistent caching
            self.shapes = None
            self.corrs = {}
        if self.correspondences:
            self._preload_correspondences()

    def _preload_correspondences(self):
        """Preload only correspondences (lightweight)."""
        for filename in self.shape_files:
            base_name, _ = os.path.splitext(filename)
            corr_filename = base_name + ".vts"
            corr_path = os.path.join(self.dataset_dir, "corr", corr_filename)
            if os.path.exists(corr_path):
                self.corrs[filename] = np.loadtxt(corr_path).astype(np.int32) - 1
            else:
                self.corrs[filename] = None

    def _preload_correspondences(self):
        """Preload only correspondences (lightweight)."""
        for filename in self.shape_files:
            base_name, _ = os.path.splitext(filename)
            corr_filename = base_name + ".vts"
            if self.correspondences:
                corr_path = os.path.join(self.dataset_dir, "corr", corr_filename)
                if os.path.exists(corr_path):
                    self.corrs[filename] = np.loadtxt(corr_path).astype(np.int32) - 1
                else:
                    self.corrs[filename] = None

    def _preload_all_shapes(self):
        """Original preloading behavior."""
        for filename in self.shape_files:
            filepath = os.path.join(self.shape_dir, filename)
            shape = get_cached_shape_data(
                filepath,
                self.shape_type,
                self.spectral,
                self.k,
                self.cache_dir,
                self.overwrite_cache,
            )

            # Move to device
            self._move_shape_to_device(shape)
            self.shapes[filename] = shape

            # Handle correspondences
            base_name, _ = os.path.splitext(filename)
            if self.correspondences and filename not in self.corrs:
                self.corrs[filename] = np.arange(shape.vertices.shape[0])

    def _get_shape(self, filename):
        """Get shape from file with caching support."""
        if self.preload_all:
            shape = self.shapes[filename]
            self._move_shape_to_device(shape)
            return shape

        filepath = os.path.join(self.shape_dir, filename)
        shape = get_cached_shape_data(
            filepath,
            self.shape_type,
            self.spectral,
            self.k,
            self.cache_dir,
            self.overwrite_cache,
        )

        # Move to device and handle correspondences
        self._move_shape_to_device(shape)

        if self.correspondences and self.corrs.get(filename) is None:
            self.corrs[filename] = np.arange(shape.vertices.shape[0])

        return shape

    def _move_shape_to_device(self, shape):
        """Move shape data to device."""
        shape.vertices = xgs.to_device(shape.vertices, self.device)
        if self.spectral:
            shape.basis.full_vals = xgs.to_device(shape.basis.full_vals, self.device)
            shape.basis.full_vecs = xgs.to_device(shape.basis.full_vecs, self.device)
        shape.laplacian._mass_matrix = xgs.to_device(
            shape.laplacian._mass_matrix, self.device
        )
        if self.shape_type == "mesh":
            shape.faces = xgs.to_device(shape.faces, self.device)

    def __getitem__(self, idx):
        """Retrieve a data sample by index."""
        filename = self.shape_files[idx]
        shape = self._get_shape(filename)

        shape_data = {"shape": shape}

        if self.correspondences:
            shape_data["corr"] = gs.array(self.corrs[filename])

        if self.distances:
            # Handle distance matrices (could also be cached similarly)
            mat_subfolder = os.path.join(self.dataset_dir, "dist")
            base_name, _ = os.path.splitext(filename)
            mat_filename = base_name + ".mat"
            dist_path = os.path.join(mat_subfolder, mat_filename)

            if os.path.exists(dist_path):
                mat_contents = scipy.io.loadmat(dist_path)
                geod_distance_matrix = mat_contents.get("D")
            else:
                # Compute and cache distance matrix
                if self.shape_type == "mesh":
                    metric = ScipyGraphShortestPathMetric(shape)
                else:
                    metric = VertexEuclideanMetric(shape)
                geod_distance_matrix = metric.dist_matrix()

                os.makedirs(os.path.dirname(dist_path), exist_ok=True)
                scipy.io.savemat(dist_path, {"D": gs.to_numpy(geod_distance_matrix)})

            shape_data["dist_matrix"] = gs.array(geod_distance_matrix)

        return shape_data

    def __len__(self):
        """Get the length of the dataset."""
        return len(self.shape_files)

    def clear_cache(self):
        """Clear the persistent cache."""
        if os.path.exists(self.cache_dir):
            import shutil

            shutil.rmtree(self.cache_dir)
            print(f"Cleared cache directory: {self.cache_dir}")


# Convenience classes for backward compatibility
class MeshDataset(ShapeDataset):
    """ShapeDataset for loading and preprocessing mesh data."""

    def __init__(
        self,
        dataset_dir,
        spectral=False,
        distances=False,
        correspondences=True,
        k=200,
        device=None,
        cache_dir=None,
        overwrite_cache=False,
        preload_all=False,
    ):
        super().__init__(
            dataset_dir=dataset_dir,
            shape_type="mesh",
            spectral=spectral,
            distances=distances,
            correspondences=correspondences,
            k=k,
            device=device,
            cache_dir=cache_dir,
            overwrite_cache=overwrite_cache,
            preload_all=preload_all,
        )


class PointCloudDataset(ShapeDataset):
    """ShapeDataset for loading and preprocessing point cloud data."""

    def __init__(
        self,
        dataset_dir,
        spectral=False,
        distances=False,
        correspondences=True,
        k=200,
        device=None,
        cache_dir=None,
        overwrite_cache=False,
        preload_all=False,
    ):
        super().__init__(
            dataset_dir=dataset_dir,
            shape_type="pointcloud",
            spectral=spectral,
            distances=distances,
            correspondences=correspondences,
            k=k,
            device=device,
            cache_dir=cache_dir,
            overwrite_cache=overwrite_cache,
            preload_all=preload_all,
        )


class PairsDataset(Dataset):
    """
    Dataset of pairs of shapes.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset or list
        Preloaded dataset or list of shape data objects.
    pair_mode : str, optional
        Strategy to generate pairs. Options: 'all', 'random'. Default is 'all'.
    n_pairs : int, optional
        Number of random pairs to generate if pair_mode is 'random'. Default is 100.
    device : torch.device, optional
        Device to move the data to. If None, uses CUDA if available, else CPU.
    """

    def __init__(self, dataset=None, pair_mode="all", pairs_ratio=100, device=None):
        # Preload meshes
        self.shape_data = dataset
        self.pair_mode = pair_mode
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Depending on pair_mode, choose the appropriate strategy
        if pair_mode == "all":
            self.pairs = self.generate_all_pairs()
        elif pair_mode == "random":
            self.pairs = self.generate_random_pairs(
                pairs_ratio
            )  # You can specify the number of pairs
        else:
            raise ValueError(f"Unsupported pair_mode: {pair_mode}")

    def generate_all_pairs(self):
        """Generate all possible pairs of shapes."""
        return list(itertools.permutations(range(self.shape_data.__len__()), 2))

    def generate_random_pairs(self, pairs_ratio=0.5):
        """Generate random pairs of shapes.

        Parameters
        ----------
        pairs_ratio : float
            Ratio of pairs to generate compared to the total number of possible pairs.
            Default is 0.5, meaning half of the possible pairs will be generated.
        """
        return random.sample(
            list(itertools.combinations(range(self.shape_data.__len__()), 2)),
            int(self.shape_data.__len__() * pairs_ratio),
        )

    def __getitem__(self, idx):
        """Get item by index.

        Parameters
        ----------
        idx : int
            Index of the item to retrieve.

        Returns
        -------
        data: dict
            Dictionary containing the source and target shapes.
        """
        src_idx, tgt_idx = self.pairs[idx]

        return {"source": self.shape_data[src_idx], "target": self.shape_data[tgt_idx]}

    def __len__(self):
        """Get the length of the dataset."""
        return len(self.pairs)
