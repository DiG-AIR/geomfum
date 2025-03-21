import torch
import os
from torch.utils.data import Dataset
import itertools
import random
from geomfum.shape.mesh import TriangleMesh
import numpy as np

class ShapeDataset(Dataset):
    def __init__(self, shape_dir, spectral=True, distances=False, k=30,device=None):
        """
        ShapeDataset
        
        Generate a dataset of shapes with notable features
        
        Args:
            shape_dir (str): Path to the directory containing the shapes.
            spectral (bool): Whether to compute the spectral features. (default True)
            k (int): Number of eigenvectors to use for the spectral features. (default 30)
            device (torch.device): Device to move the data to.
        """

        self.shape_dir = shape_dir
        self.shape_files = sorted([f for f in os.listdir(shape_dir) if f.endswith('.off')])    # off but we can accept also otherkind of files TODO: generalize

        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.spectral = spectral
        self.k = k
        self.distances = distances
        # Preload meshes (or their important features) into memory
        self.meshes = {}
        self.corrs = {}
        for filename in self.shape_files:
            mesh = TriangleMesh.from_file(os.path.join(self.shape_dir, filename))
            if spectral:
                mesh.laplacian.find_spectrum(spectrum_size=self.k, set_as_basis=True)
                mesh.basis.use_k = self.k
            if distances:
                mesh.geod_distance_matrix
            
            self.meshes[filename] = mesh
            corr_filename = filename.replace('.off', '.vts')  # Assuming correspondence files are .txt
            if os.path.exists(os.path.join(self.shape_dir, 'corr', corr_filename)):
                self.corrs[filename] = np.loadtxt(os.path.join(self.shape_dir, 'corr', corr_filename)).astype(np.int32)
            else:
                self.corrs[filename] = np.arange(mesh.vertices.shape[0])
    def __getitem__(self, idx):

        filename = self.shape_files[idx]
        mesh = self.meshes[filename]
        
        mesh.to_torch(self.device)
        # the datas are stored in dictionaries
        data = {
            'vertices': mesh.vertices,
            'faces': mesh.faces,
            'corr': self.corrs[filename]
        }
        if self.spectral:
            mesh.use_k=self.k
            mesh.basis.to_torch(self.device)
            data.update({
                'evals': mesh.basis.vals,
                'basis': mesh.basis.vecs,
                'pinv': mesh.basis.pinv
            })
        if self.distances:
            data.update({
                'distances': mesh.geod_distance_matrix
            })
        
        return data

    def __len__(self):
        return len(self.shape_files)

    
class PairsDataset(Dataset):
    def __init__(self, dataset=None ,pair_mode='all',n_pairs=100, device=None):
        """
        Dataset of pairs of shapes.
        Args:
            shape_dir (str): Path to the directory containing the shapes.
            pair_mode (str): Strategy to generate pairs. Options: 'all', 'random', 'category_based'. (default 'all')
            spectral (bool): Whether to compute the spectral features. (default True)
            k (int): Number of eigenvectors to use for the spectral features. (default 30)
            device (torch.device): Device to move the data to.
        """
        
        # Preload meshes
        self.shape_data= dataset
        self.pair_mode = pair_mode
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Depending on pair_mode, choose the appropriate strategy
        if pair_mode == 'all':
            self.pairs = self.generate_all_pairs()
        elif pair_mode == 'random':
            self.pairs = self.generate_random_pairs(n_pairs)  # You can specify the number of pairs
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
    
    
