"""
This file contains the implementation of the deep functional map network approach.
In 'functional_map.py' we defined the energies and the optimization problem for the functional map.
At the same time here we define the loss functions to optimize a functional map.


In Deep Functional Maps, the fiunctional map is computed by the forward pass computed on given descriptors.
The algorithm that performs this pass is typically called FunctionalMapNet
"""

#we define an abstract class for the forward pass. Depending on the choices, the forward pass can be implemented in different ways

# TODO: Add bidirectionality parameter

import torch
import torch.nn as nn
import torch.functional as F
import abc


class ForwardFunctionalMap(nn.Module):
    def __init__(self, lmbda=0, resolvant_gamma=1):
        super(ForwardFunctionalMap, self).__init__()
        """Class for the forward pass of the functional map
        lmbda (float): weight of the mask
        resolvant_gamma (float): resolvant of the regularized functional map as in 
        proper (bool): if True, the functional map is converted into a proper functional map
        "Structured Regularization of Functional Map Computations, Jing Ren, Mikhail Panine, Peter Wonka, Maks Ovsjanikov, SGP 2019"
        """
        self.lmbda = lmbda
        self.resolvant_gamma = resolvant_gamma
    def forward(self, mesh_x, mesh_y, feat_x, feat_y):
        """
        Forward pass to compute functional map
        Args:
            feat_x (torch.Tensor): feature vector of shape x. [B, Vx, C].
            feat_y (torch.Tensor): feature vector of shape y. [B, Vy, C].

        Returns:
            C (torch.Tensor): functional map from shape x to shape y. [B, K, K].
        """
        device=mesh_x.vertices.device
        k1=mesh_x.basis.use_k
        k2=mesh_y.basis.use_k
                
        #for the moment we need to transpose the features
        if feat_x.dim()==2:
            feat_x= torch.tensor(feat_x.T)[None].to(device)
            feat_y= torch.tensor(feat_y.T)[None].to(device)
        else:
            feat_x= feat_x.to(torch.float32)
            feat_y= feat_y.to(torch.float32)
        
        #load evals and evecs adn pinv
        
        evals_x = mesh_x.basis.vals[:k1][None].to(torch.float32).to(device)
        evals_y = mesh_y.basis.vals[:k2][None].to(torch.float32).to(device)
        
        evecs_trans_x = mesh_x.basis.pinv[:k1,:][None].to(torch.float32).to(device)
        evecs_trans_y = mesh_y.basis.pinv[:k2,:][None].to(torch.float32).to(device)

        if self.lmbda>0:
            MASK = self.get_mask(evals_x, evals_y, self.resolvant_gamma)  # [B, K, K]
        A_x = torch.bmm(evecs_trans_x, feat_x)  # [B, K, C]
        A_y = torch.bmm(evecs_trans_y, feat_y)  # [B, K, C]

        A_x_t= A_x.transpose(1,2)
            
        AA_xx = torch.bmm(A_x, A_x_t)  # [B, K, K]
        AA_yx = torch.bmm(A_y, A_x_t)  # [B, K, K]

        C_i = []
        for i in range(evals_x.shape[1]):
            if self.lmbda==0:
                C = torch.bmm(torch.inverse(AA_xx ), AA_yx[:, [i], :].transpose(1, 2))
            else:
                MASK_i = torch.cat([torch.diag(MASK[bs, i, :].flatten()).unsqueeze(0) for bs in range(evals_x.shape[0])], dim=0)
                
                C = torch.bmm(torch.inverse(AA_xx + self.lmbda * MASK_i), AA_yx[:, [i], :].transpose(1, 2))
            C_i.append(C.transpose(1, 2))   
        
        Cxy = torch.cat(C_i, dim=1)
        
        return Cxy

    def _compute_mask(self,evals1, evals2, resolvant_gamma):
        """Compute the mask for the functional map in bathc"""

        scaling_factor = max(torch.max(evals1), torch.max(evals2))
        evals1, evals2 = evals1 / scaling_factor, evals2 / scaling_factor
        evals_gamma1 = (evals1 ** resolvant_gamma)[None, :]
        evals_gamma2 = (evals2 ** resolvant_gamma)[:, None]

        M_re = evals_gamma2 / (evals_gamma2.square() + 1) - evals_gamma1 / (evals_gamma1.square() + 1)
        M_im = 1 / (evals_gamma2.square() + 1) - 1 / (evals_gamma1.square() + 1)
        return M_re.square() + M_im.square()


    def get_mask(self,evals1, evals2, resolvant_gamma):
        """Compute the mask for the functional map in bathc"""
        masks = []
        for bs in range(evals1.shape[0]):
            masks.append(self._compute_mask(evals1[bs], evals2[bs], resolvant_gamma))
        return torch.stack(masks, dim=0)


