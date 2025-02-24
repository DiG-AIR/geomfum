import torch
import os
from torch.utils.data import Dataset
import itertools
import random
from geomfum.shape.mesh import TriangleMesh

class ShapeDataset(Dataset):
    def __init__(self, shape_dir, device=None):
        self.shape_dir = shape_dir
        self.shape_files = sorted([f for f in os.listdir(shape_dir) if f.endswith('.off')])    # off but we can accept also otherkind of files
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Preload meshes (or their important features) into memory
        self.meshes = {}
        for filename in self.shape_files:
            mesh = TriangleMesh.from_file(os.path.join(self.shape_dir, filename))
            mesh.laplacian.find_spectrum(spectrum_size=40, set_as_basis=True)
            mesh.basis.use_k = 30
            self.meshes[filename] = mesh
    def __getitem__(self, idx):
        # Retrieve the pair of filenames
        filename = self.shape_files[idx]
        
        # Retrieve preloaded meshes
        mesh = self.meshes[filename]
        mesh.use_k=30
        # Convert meshes to tensors and move to device (for training)
        mesh.to_torch(self.device)
    
        mesh.basis.to_torch(self.device)
        # Create a dictionary with all the relevant data
        
        data = {
            'vertices': mesh.vertices,
            'faces': mesh.faces,
            'evals': mesh.basis.vals,
            'basis': mesh.basis.vecs,
            'pinv': mesh.basis.pinv
        }
        return data

    def __len__(self):
        return len(self.shape_files)

    
class PairsDataset(Dataset):
    def __init__(self, shape_dir, pair_mode='all', device=None):
        self.shape_dir = shape_dir
        # Preload meshes
        self.shape_data= ShapeDataset(shape_dir, device=device)
        self.pair_mode = pair_mode
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Depending on pair_mode, choose the appropriate strategy
        if pair_mode == 'all':
            self.pairs = self.generate_all_pairs()
        elif pair_mode == 'random':
            self.pairs = self.generate_random_pairs(n_pairs=100)  # You can specify the number of pairs
        else:
            raise ValueError(f"Unsupported pair_mode: {pair_mode}")

    def generate_all_pairs(self):
        """Generate all possible pairs of shapes."""
        return list(itertools.combinations(range(self.shape_data.__len__()), 2))

    def generate_random_pairs(self, n_pairs=100):
        """Generate random pairs of shapes."""
        return random.sample(list(itertools.combinations(range(self.shape_data.__len__()), 2)), n_pairs)

    def generate_category_based_pairs(self, category_dict):
        """Generate pairs based on a specific category."""
        pairs = []
        for category, filenames in category_dict.items():
            pairs.extend(itertools.combinations(range(self.shape_data.__len__()), 2))
        return pairs

    def __getitem__(self, idx):
        # Retrieve the pair of filenames
        src_idx, tgt_idx = self.pairs[idx]
        
        return {'source':self.shape_data[src_idx], 'target':self.shape_data[tgt_idx]}

    def __len__(self):
        return len(self.pairs)
    
    
