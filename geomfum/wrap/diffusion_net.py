"""
This is the wrapper of the diffusion net model.
#TODO: Add references
#TODO: For the moment we assume to have the implementation of Diffusionet somewhere in the code
"""

from geomfum.feature_extractor.diffusion_net.diffusion_network import DiffusionNet
from geomfum.descriptor import Descriptor
import torch

class DiffusionNetDescriptor(Descriptor):
    """Descriptor representing the output of DiffusionNet."""

    def __init__(self, in_channels=3, out_channels=128, hidden_channels=128, n_block=4, last_activation=None, 
                 mlp_hidden_channels=None, output_at='vertices', dropout=True, with_gradient_features=True, 
                 with_gradient_rotations=True, diffusion_method='spectral', k_eig=128, cache_dir=None, 
                 input_type='xyz'):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.n_block = n_block
        self.last_activation = last_activation
        self.mlp_hidden_channels = mlp_hidden_channels
        self.output_at = output_at
        self.dropout = dropout
        self.with_gradient_features = with_gradient_features
        self.with_gradient_rotations = with_gradient_rotations
        self.diffusion_method = diffusion_method
        self.k_eig = k_eig
        self.cache_dir = cache_dir
        self.input_type = input_type
        self.model = DiffusionNet(in_channels=self.in_channels, out_channels=self.out_channels, 
                                  hidden_channels=self.hidden_channels, n_block=self.n_block, 
                                  last_activation=self.last_activation, mlp_hidden_channels=self.mlp_hidden_channels, 
                                  output_at=self.output_at, dropout=self.dropout, 
                                  with_gradient_features=self.with_gradient_features, 
                                  with_gradient_rotations=self.with_gradient_rotations, 
                                  diffusion_method=self.diffusion_method, k_eig=self.k_eig, 
                                  cache_dir=self.cache_dir, input_type=self.input_type)

        self.n_features = self.out_channels

    def __call__(self,mesh):

        #TODO: add to convert the mesh to tensor
        v=torch.tensor(mesh.vertices)[None].to(torch.float32)
        f=torch.tensor(mesh.faces)[None].to(torch.int32)
        self.features = self.model(v,f)
        return self.features
    
    def load(self,path):
        #load model parameters from the provided path
        self.model.load_state_dict(torch.load(path))