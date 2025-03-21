"""
This file contains the implementation of the deep functional map network approach.
In 'functional_map.py' we defined the energies and the optimization problem for the functional map.
At the same time here we define the loss functions to optimize a functional map.

In Deep Functional Maps, the fiunctional map is computed by the forward pass computed on given descriptors.
The algorithm that performs this pass is typically called FunctionalMapNet
"""

import torch
import torch.nn as nn
import torch.functional as F
import abc
from geomfum.shape.mesh import TriangleMesh

class ForwardFunctionalMap(nn.Module):
    def __init__(self, lmbda=0, resolvent_gamma=1, bijective=True):
        super(ForwardFunctionalMap, self).__init__()
        """Class for the forward pass of the functional map.
        Args:
            lmbda (float): weight of the mask
            resolvant_gamma (float): resolvant of the regularized functional map.
            bijective (bool): whether we compute the map inboth the directions.
        """
        self.lmbda = lmbda
        self.resolvent_gamma = resolvent_gamma
        self.bijective = bijective

    def compute_functional_map(self, feat_a, feat_b, evals_a, evals_b, pinv_a, pinv_b):
        """Compute the functional map between two shapes.
        Args:
            feat_a (torch.Tensor): Feature vector of shape a. [B, Vx, C].
            feat_b (torch.Tensor): Feature vector of shape b. [B, Vy, C].
            evals_a (torch.Tensor): Eigenvalues of shape a. [B, K].
            evals_b (torch.Tensor): Eigenvalues of shape b. [B, K].
        """
        # Compute the functional map (C)
        if self.lmbda > 0:
            MASK = self.get_mask(evals_a, evals_b, self.resolvent_gamma)  # [B, K, K]
        
        A_a = torch.bmm(pinv_a, feat_a)  # [B, K, C]
        A_b = torch.bmm(pinv_b, feat_b)  # [B, K, C]

        A_a_t = A_a.transpose(1, 2)

        AA_aa = torch.bmm(A_a, A_a_t)  # [B, K, K]
        AA_ba = torch.bmm(A_b, A_a_t)  # [B, K, K]

        C_i = []
        for i in range(evals_a.shape[1]):
            if self.lmbda == 0:
                C = torch.bmm(torch.inverse(AA_aa), AA_aa[:, [i], :].transpose(1, 2))
            else:
                MASK_i = torch.cat([torch.diag(MASK[bs, i, :].flatten()).unsqueeze(0) for bs in range(evals_a.shape[0])], dim=0)
                C = torch.bmm(torch.inverse(AA_aa + self.lmbda * MASK_i), AA_ba[:, [i], :].transpose(1, 2))
            C_i.append(C.transpose(1, 2))   
        
        Cab = torch.cat(C_i, dim=1)
        return Cab
        

    def forward(self, mesh_a, mesh_b, feat_a, feat_b):
        """
        Forward pass to compute functional map.
        Args:
            mesh_a: A TriangleMesh object or a dictionary containing mesh data for shape a.
            mesh_b: A TriangleMesh object or a dictionary containing mesh data for shape b.
            feat_a (torch.Tensor): Feature vector of shape a. [B, Vx, C].
            feat_b (torch.Tensor): Feature vector of shape b. [B, Vy, C].
        Returns:
            Cab (torch.Tensor): Functional map from shape a to shape b. [B, K, K].
            Cba (torch.Tensor): Functional map from shape a to shape b. [B, K, K].

        """
        device = feat_a.device
        
        # Handle mesh_a (TriangleMesh or dictionary)
        if isinstance(mesh_a, dict):
            k1 = mesh_a['basis'].shape[-1]
            evals_a = mesh_a['evals'].to(torch.float32)
            pinv_a = mesh_a['pinv'].to(torch.float32).to(device)
        elif isinstance(mesh_a, TriangleMesh):
            k1 = mesh_a.basis.use_k
            evals_a = mesh_a.basis.vals[:k1][None].to(torch.float32).to(device)
            pinv_a = mesh_a.basis.pinv[:k1, :][None].to(torch.float32).to(device)
        else:
            raise TypeError("mesh_a must be either a TriangleMesh or a dictionary containing 'vertices', 'faces', and 'basis'.")

        if isinstance(mesh_b, dict):
            k2 = mesh_b['basis'].shape[-1]
            evals_b = mesh_b['evals'].to(torch.float32)
            pinv_b = mesh_b['pinv'].to(torch.float32).to(device)
        elif isinstance(mesh_b, TriangleMesh):
            k2 = mesh_b.basis.use_k
            evals_b = mesh_b.basis.vals[:k2][None].to(torch.float32).to(device)
            pinv_b = mesh_b.basis.pinv[:k2, :][None].to(torch.float32).to(device)
        else:
            raise TypeError("mesh_b must be either a TriangleMesh or a dictionary containing 'vertices', 'faces', and 'basis'.")

        # Prepare feature tensors (ensure they're in the right format)
        if evals_a.dim() == 1: evals_a = evals_a.unsqueeze(0)
        if evals_b.dim() == 1:  evals_b = evals_b.unsqueeze(0)
        if pinv_a.dim() == 2:  pinv_a = pinv_a.unsqueeze(0)
        if pinv_b.dim() == 2:  pinv_b = pinv_b.unsqueeze(0)

        if isinstance(feat_a,torch.Tensor):
            if feat_a.dim() == 2: feat_a = feat_a.unsqueeze(0).to(device)
            if feat_b.dim() == 2: feat_b = feat_b.unsqueeze(0).to(device)
        else:
            feat_a = torch.tensor(feat_a.T).unsqueeze(0).to(device)
            feat_b = torch.tensor(feat_b.T).unsqueeze(0).to(device)
        
        Cab = self.compute_functional_map(feat_a, feat_b, evals_a, evals_b, pinv_a, pinv_b)
        
        if self.bijective:
            Cba = self.compute_functional_map(feat_b, feat_a, evals_b, evals_a, pinv_b, pinv_a)
        else:
            Cba = None
        return Cab, Cba

    #TODO: Uniform these functions
    def _compute_mask(self, evals_a, evals_b, resolvant_gamma):
        """Compute the mask for the functional map in batch."""
        scaling_factor = max(torch.max(evals_a), torch.max(evals_b))
        evals_a, evals_b = evals_a / scaling_factor, evals_b / scaling_factor
        evals_gamma_a = (evals_a ** resolvant_gamma)[None, :]
        evals_gamma_b = (evals_b ** resolvant_gamma)[:, None]

        M_re = evals_gamma_b / (evals_gamma_b.square() + 1) - evals_gamma_a / (evals_gamma_a.square() + 1)
        M_im = 1 / (evals_gamma_b.square() + 1) - 1 / (evals_gamma_a.square() + 1)
        return M_re.square() + M_im.square()


    def get_mask(self, evals_a, evals_b, resolvant_gamma):
        """Compute the mask for the functional map in batch."""
        masks = []
        for bs in range(evals_a.shape[0]):
            masks.append(self._compute_mask(evals_a[bs], evals_b[bs], resolvant_gamma))
        return torch.stack(masks, dim=0)
