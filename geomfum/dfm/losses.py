"""
This file contains the implementation of useful loss functions for deep functional map.
They are organized with a loss managaer and registered by a Lossregistry.
"""

import torch
import torch.nn as nn
from geomfum._registry import LossRegistry

class LossManager:
    def __init__(self, loss_configs):
        """
        loss Manager: Dictionary where keys are loss names, and values are their weights.
        Inputs:
            - Loss_configs: Dictionary of loss names and weights.
        """
        self.losses = {
            name: (LossRegistry.get(name), weight) for name, weight in loss_configs.items()
        }

    def compute_loss(self, outputs):
        total_loss = 0
        loss_dict = {}
        
        for loss_name, (loss_fn, weight) in self.losses.items():
            required_inputs = {key: outputs[key] for key in loss_fn.required_inputs}
            loss_value = loss_fn(**required_inputs) * weight
            loss_dict[loss_name] = loss_value.item()
            total_loss += loss_value

        return total_loss, loss_dict


######################LOSS IMPLEMENTATIONS ############################

class SquaredFrobeniusLoss(nn.Module):
    """
    Compute the distance induced by the frobenius norm between two vectors/matrices
    Inputs: 
    - a: First vector/matrix
    - b: Second vector/matrix
    """
    def forward(self, a, b):
        return torch.mean(torch.sum(torch.abs(a - b) ** 2, dim=(-2, -1)))

class OrthonormalityLoss(nn.Module):
    """
    Computes the Orthonormality error of a functional map  \| C^T C - I \|
    Inputs: 
    - fmap: Functional map
    """    
    def __init__(self, weight=1):
        super().__init__()
        self.weight = weight

    required_inputs = ["Cab"]            
    def forward(self, Cab):
        metric = SquaredFrobeniusLoss()
        eye = torch.eye(Cab.shape[1], device=Cab.device).unsqueeze(0).expand(Cab.shape[0], -1, -1)
        return self.weight * metric(torch.bmm(Cab.transpose(1, 2), Cab), eye)


class BijectivityLoss(nn.Module):
    """
    Computes the Bijectivity error of two functional maps \| Cab Cba - I  \|
    Inputs:
    - Cab: Functional map from shape 1 to shape 2
    - Cba: Functional map from shape 2 to shape 1
    """
    def __init__(self, weight=1):
        super().__init__()
        self.weight = weight

    required_inputs = ["Cab", "Cba"]
    def forward(self, Cab, Cba):
        metric = SquaredFrobeniusLoss()
        eye = torch.eye(Cab.shape[1], device=Cab.device).unsqueeze(0).expand(Cab.shape[0], -1, -1)
        return self.weight * metric(torch.bmm(Cab, Cba), eye) + metric(torch.bmm(Cba, Cab), eye)


class LaplacianCommutativityLoss(nn.Module):
    """
    Computes the Laplacian Commutativity error of a functional map
    Inputs:
    - fmap: Functional map
    - evals_a: Eigenvalues of the first shape
    - evals_b: Eigenvalues of the second shape
    """
    def __init__(self, weight=1):
        super().__init__()
        self.weight = weight

    required_inputs = ["Cab", "source", "target"]
    def forward(self, Cab, source,target):
        metric = SquaredFrobeniusLoss()
        return self.weight * metric(torch.einsum('abc,ac->abc', Cab, source['evals']), torch.einsum('ab,abc->abc', target['evals'], Cab))


class Fmap_Supervision(nn.Module):
    """WW
    Computes the Laplacian Commutativity error of a functional map
    Inputs:
    - fmap: Functional map
    """
    def __init__(self, weight=1):
        super().__init__()
        self.weight = weight

    required_inputs = ["Cab", "Cab_sup"]
    def forward(self, Cab, Cab_sup):
        metric = SquaredFrobeniusLoss()
        return self.weight * metric(Cab,Cab_sup)

