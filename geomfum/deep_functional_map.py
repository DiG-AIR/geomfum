"""
This file contains the implementation of the deep functional map network approach.
In 'functional_map.py' we defined the energies and the optimization problem for the functional map.
At the same time here we define the loss functions to optimize a functional map.


In Deep Functional Maps, the fiunctional map is computed by the forward pass computed on given descriptors.
The algorithm that performs this pass is typically called FunctionalMapNet
"""

#we define an abstract class for the forward pass. Depending on the choices, the forward pass can be implemented in different ways

import torch
import torch.nn as nn
import abc



class ForwardFunctionalMap(nn.Module):
    """Abstract class for the forward pass of the functional map"""
    @abc.abstractmethod
    def forward(self, feat_x, feat_y):
        """
        Forward pass to compute functional map
        Args:
            feat_x (torch.Tensor): feature vector of shape x. [B, Vx, C].
            feat_y (torch.Tensor): feature vector of shape y. [B, Vy, C].

        Returns:
            C (torch.Tensor): functional map from shape x to shape y. [B, K, K].
        """
        pass



class RegularizedFMNet(ForwardFunctionalMap):
    """Compute the functional map matrix representation in DPFM"""
    def __init__(self, lmbda=100, resolvant_gamma=0.5):
        super(RegularizedFMNet, self).__init__()
        self.lmbda = lmbda
        self.resolvant_gamma = resolvant_gamma

    def forward(self, mesh_x, mesh_y, feat_x, feat_y):

        k1=mesh_x.basis.use_k
        k2=mesh_y.basis.use_k

        feat_x= torch.tensor(feat_x.T)[None]
        feat_y= torch.tensor(feat_y.T)[None]

        evals_x, evecs_x = torch.tensor(mesh_x.basis.vals[:k1])[None], torch.tensor(mesh_x.basis.vecs[:,:k1])[None]
        evals_y, evecs_y = torch.tensor(mesh_y.basis.vals[:k2])[None], torch.tensor(mesh_y.basis.vecs[:,:k2])[None]

        evecs_trans_x = torch.transpose(evecs_x,2,1)
        evecs_trans_y = torch.transpose(evecs_y, 2, 1)
        A = torch.bmm(evecs_trans_x, feat_x)  # [B, K, C]
        B = torch.bmm(evecs_trans_y, feat_y)  # [B, K, C]

        D = self.get_mask(evals_x, evals_y, self.resolvant_gamma)  # [B, K, K]
        A_t = A.transpose(1, 2)  # [B, C, K]
        A_A_t = torch.bmm(A, A_t)  # [B, K, K]
        B_A_t = torch.bmm(B, A_t)  # [B, K, K]

        C_i = []
        for i in range(evals_x.shape[1]):
            D_i = torch.cat([torch.diag(D[bs, i, :].flatten()).unsqueeze(0) for bs in range(evals_x.shape[0])], dim=0)
            C = torch.bmm(torch.inverse(A_A_t + self.lmbda * D_i), B_A_t[:, [i], :].transpose(1, 2))
            C_i.append(C.transpose(1, 2))

        Cxy = torch.cat(C_i, dim=1)
        return Cxy


    #this function returns a mask that is used to regularize the functional map
    #TODO: CHANGE THESE FUNCTIONS
    def _get_mask(self,evals1, evals2, resolvant_gamma):
        scaling_factor = max(torch.max(evals1), torch.max(evals2))
        evals1, evals2 = evals1 / scaling_factor, evals2 / scaling_factor
        evals_gamma1 = (evals1 ** resolvant_gamma)[None, :]
        evals_gamma2 = (evals2 ** resolvant_gamma)[:, None]

        M_re = evals_gamma2 / (evals_gamma2.square() + 1) - evals_gamma1 / (evals_gamma1.square() + 1)
        M_im = 1 / (evals_gamma2.square() + 1) - 1 / (evals_gamma1.square() + 1)
        return M_re.square() + M_im.square()


    def get_mask(self,evals1, evals2, resolvant_gamma):
        masks = []
        for bs in range(evals1.shape[0]):
            masks.append(self._get_mask(evals1[bs], evals2[bs], resolvant_gamma))
        return torch.stack(masks, dim=0)




