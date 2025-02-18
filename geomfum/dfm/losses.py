"""
This code contains implementations of useful losses to train deep functional maps.
we inherit the structure of FactorSum from the the funtional_map file
"""


import torch
import torch.nn as nn
import torch.nn.functional as F




class LossWeightedFactor(nn.Module):
    """Weighted factor.

    Parameters
    ----------
    weight : float
        Weight of the factor.
    """

    def __init__(self, weight):
        self.weight = weight

    @nn.abstractmethod
    def forward(self, fmap_matrix):
        """Compute energy.

        Parameters
        ----------
        fmap_matrix : array-like, shape=[spectrum_size_b, spectrum_size_a]
            Functional map matrix.

        Returns
        -------
        weighted_energy : float
            Weighted energy associated with the factor.
        """
        pass

class LossFactorSum(LossWeightedFactor):
    """Factor sum.

    Parameters
    ----------
    factors : list[WeightedFactor]
        Factors.
    """

    def __init__(self, factors, weight=1.0):
        super().__init__(weight=weight)
        self.factors = factors

    def forward(self, fmap_matrix):
        """Compute energy.

        Parameters
        ----------
        fmap_matrix : array-like, shape=[spectrum_size_b, spectrum_size_a]
            Functional map matrix.

        Returns
        -------
        weighted_energy : float
            Weighted energy associated with the factor.
        """
        return self.weight * torch.sum([factor(fmap_matrix) for factor in self.factors])


# This is a metric that we can change so maybe we can intranstiate it somewhere else
class SquaredFrobeniusLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, a, b):
        loss = torch.sum(torch.abs(a - b) ** 2, dim=(-2, -1))
        return self.loss_weight * torch.mean(loss)

class OrthonormalityLoss(LossWeightedFactor):
    def __init__(self, weight=1.0):
        super().__init__(weight=weight)
        self.metric=SquaredFrobeniusLoss()
    def forward(self, fmap):
        eye = torch.eye(fmap.shape[1], fmap.shape[2], device=fmap.device).unsqueeze(0)
        eye_batch = torch.repeat_interleave(eye, repeats=fmap.shape[0], dim=0)

        return   self.weight * self.metric(torch.bmm(fmap.transpose(1, 2), fmap), eye_batch)


class BijectivityLoss(LossWeightedFactor):
    def __init__(self, weight=1.0):
        super().__init__(weight=weight)
        self.metric=SquaredFrobeniusLoss()
        
    def forward(self, fmap12, fmap21):
        eye = torch.eye(fmap12.shape[1], fmap12.shape[2], device=fmap12.device).unsqueeze(0)
        eye_batch = torch.repeat_interleave(eye, repeats=fmap12.shape[0], dim=0)

        return   self.weight * (self.metric(torch.bmm(fmap12, fmap21), eye_batch)+self.metric(torch.bmm(fmap21, fmap12), eye_batch))

class LaplacianCommutativityLoss(LossWeightedFactor):
    def __init__(self, weight=1.0):
        super().__init__(weight=weight)
        self.metric=SquaredFrobeniusLoss()
        
    def forward(self, fmap, evals1, evals2):

        return self.weight * self.metric(torch.einsum('abc,ac->abc', fmap, evals1),  torch.einsum('ab,abc->abc', evals2, fmap))
