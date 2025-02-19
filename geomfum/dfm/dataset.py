import torch
import os
from torch.utils.data import Dataset
import itertools
import random
from geomfum.shape.mesh import TriangleMesh

class ShapeDataset(Dataset):
    def __init__(self, shape_dir, pair_mode='all', device=None):
        self.shape_dir = shape_dir
        self.shape_files = sorted([f for f in os.listdir(shape_dir) if f.endswith('.off')])    # off but we can accept also otherkind of files
        self.pair_mode = pair_mode
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Depending on pair_mode, choose the appropriate strategy
        if pair_mode == 'all':
            self.pairs = self.generate_all_pairs()
        elif pair_mode == 'random':
            self.pairs = self.generate_random_pairs(n_pairs=100)  # You can specify the number of pairs
        else:
            raise ValueError(f"Unsupported pair_mode: {pair_mode}")

        # Preload meshes (or their important features) into memory
        self.meshes = {}
        for filename in self.shape_files:
            mesh = TriangleMesh.from_file(os.path.join(self.shape_dir, filename))
            mesh.laplacian.find_spectrum(spectrum_size=40, set_as_basis=True)
            mesh.basis.use_k = 30
            self.meshes[filename] = mesh

    def generate_all_pairs(self):
        """Generate all possible pairs of shapes."""
        return list(itertools.combinations(self.shape_files, 2))

    def generate_random_pairs(self, n_pairs=100):
        """Generate random pairs of shapes."""
        return random.sample(list(itertools.combinations(self.shape_files, 2)), n_pairs)

    def generate_category_based_pairs(self, category_dict):
        """Generate pairs based on a specific category."""
        pairs = []
        for category, filenames in category_dict.items():
            pairs.extend(itertools.combinations(filenames, 2))
        return pairs

    def __getitem__(self, idx):
        # Retrieve the pair of filenames
        filename_a, filename_b = self.pairs[idx]
        
        # Retrieve preloaded meshes
        mesh_a = self.meshes[filename_a]
        mesh_b = self.meshes[filename_b]

        # Convert meshes to tensors and move to device (for training)
        mesh_a.to_torch(self.device)
        mesh_b.to_torch(self.device)
        mesh_a.basis.to_torch(self.device)
        mesh_b.basis.to_torch(self.device)

        # Create a dictionary with all the relevant data
        data_source = {
            'vertices': mesh_a.vertices,
            'faces': mesh_a.faces,
            'evals': mesh_a.basis.vals,
            'basis': mesh_a.basis.vecs,
            'pinv': mesh_a.basis.pinv
        }
        data_target = {
            'vertices': mesh_b.vertices,
            'faces': mesh_b.faces,
            'evals': mesh_b.basis.vals,
            'basis': mesh_b.basis.vecs,
            'pinv': mesh_b.basis.pinv
        }
        return data_source, data_target

    def __len__(self):
        return len(self.pairs)