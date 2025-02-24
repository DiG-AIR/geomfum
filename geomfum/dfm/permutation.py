"""
This file contains an implementation to obtaina  permutation betwee two features

For the moment is just a wrappe rof densemesh form robin magnet  
https://arxiv.org/html/2404.00330v1
"""

from densemaps.torch import maps
import torch.nn as nn

class PermutationModule(nn.Module):
    
    def __init__(self, blur=0.1, permutation='dense'):
        super(PermutationModule, self).__init__()   
        self.param=permutation
        self.blur=blur
    def forward(self, feat1, feat2):
        P21 = maps.KernelDistMap(feat1, feat2, blur=self.blur)  # A "dense" kernel map, not used in memory

        if self.param=='dense':
            return  P21._to_dense() 
        if self.param=='index':
            return  P21.get_nn() 
        else:
            return P21 

