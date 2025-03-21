"""
This file contains an implementation to obtaina  permutation betwee two features

For the moment is just a wrappe rof densemesh form robin magnet  
https://arxiv.org/html/2404.00330v1
Memory-Scalable and Simplified Functional Map Learning, Robin Magnet, Maks Ovsjanikov 2024
"""

from densemaps.torch import maps
import torch.nn as nn

class PermutationModule(nn.Module):
    def __init__(self, tau=0.07):
        super(PermutationModule, self).__init__()   
        self.tau=tau
    def forward(self, feat_a, feat_b):
        Pba = nn.functional.softmax(feat_a@feat_b.transpose(-1,-2)/self.tau, dim=-1)

        return Pba



class DensePermutation(nn.Module):
    
    def __init__(self, blur=0.1, permutation='dense'):
        super(DensePermutation, self).__init__()   
        self.param=permutation
        self.blur=blur
    def forward(self, feat_a, feat_b):
        Pba = maps.KernelDistMap(feat_a, feat_b, blur=self.blur)  # A "dense" kernel map, not used in memory

        if self.param=='dense':
            return  Pba._to_dense() 
        if self.param=='index':
            return  Pba.get_nn() 
        else:
            return Pba

