import torch
import torch.nn as nn

class LossRegistry:
    _losses = {}

    @classmethod
    def register(cls, name):
        def decorator(loss_class):
            cls._losses[name] = loss_class
            return loss_class
        return decorator

    @classmethod
    def get(cls, name, *args, **kwargs):
        if name not in cls._losses:
            raise ValueError(f"Loss {name} not found. Available: {list(cls._losses.keys())}")
        return cls._losses[name](*args, **kwargs)


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

    def compute_loss(self, **kwargs):
        total_loss = 0
        loss_dict = {}
        

        for loss_name, (loss_fn, weight) in self.losses.items():
            required_inputs = {key: kwargs[key] for key in loss_fn.required_inputs}
            loss_value = loss_fn(**required_inputs) * weight
            loss_dict[loss_name] = loss_value.item()
            total_loss += loss_value

        return total_loss, loss_dict


######################LOSS IMPLEMENTATIONS ############################

@LossRegistry.register("Frobenius")
class SquaredFrobeniusLoss(nn.Module):
    """
    Compute the distance induced by the frobenius norm between two vectors/matrices
    Inputs: 
    - a: First vector/matrix
    - b: Second vector/matrix
    """
    def forward(self, a, b):
        return torch.mean(torch.sum(torch.abs(a - b) ** 2, dim=(-2, -1)))

@LossRegistry.register("Orthonormality")
class OrthonormalityLoss(nn.Module):
    """
    Computes the Orthonormality error of a functional map
    Inputs: 
    - fmap: Functional map
    """    
    def __init__(self, weight=1):
        super().__init__()
        self.weight = weight

    required_inputs = ["Cxy"]            
    def forward(self, Cxy):
        metric = SquaredFrobeniusLoss()
        eye = torch.eye(Cxy.shape[1], device=Cxy.device).unsqueeze(0).expand(Cxy.shape[0], -1, -1)
        return self.weight * metric(torch.bmm(Cxy.transpose(1, 2), Cxy), eye)


@LossRegistry.register("Bijectivity")
class BijectivityLoss(nn.Module):
    """
    Computes the Bijectivity error of two functional maps
    Inputs:
    - fmap12: Functional map from shape 1 to shape 2
    - fmap21: Functional map from shape 2 to shape 1
    """
    def __init__(self, weight=1):
        super().__init__()
        self.weight = weight

    required_inputs = ["Cxy", "Cyx"]
    def forward(self, Cxy, Cyx):
        metric = SquaredFrobeniusLoss()
        eye = torch.eye(Cxy.shape[1], device=Cxy.device).unsqueeze(0).expand(Cxy.shape[0], -1, -1)
        return self.weight * metric(torch.bmm(Cxy, Cyx), eye) + metric(torch.bmm(Cyx, Cxy), eye)


@LossRegistry.register("Laplacian_Commutativity")
class LaplacianCommutativityLoss(nn.Module):
    """
    Computes the Laplacian Commutativity error of a functional map
    Inputs:
    - fmap: Functional map
    - evals1: Eigenvalues of the first shape
    - evals2: Eigenvalues of the second shape
    """
    def __init__(self, weight=1):
        super().__init__()
        self.weight = weight

    required_inputs = ["Cxy", "evals_x", "evals_y"]
    def forward(self, Cxy, evals_x, evals_y):
        metric = SquaredFrobeniusLoss()
        return self.weight * metric(torch.einsum('abc,ac->abc', Cxy, evals_x), torch.einsum('ab,abc->abc', evals_y, Cxy))


@LossRegistry.register("Fmap_Supervision")
class Fmap_Supervision(nn.Module):
    """
    Computes the Laplacian Commutativity error of a functional map
    Inputs:
    - fmap: Functional map
    - evals1: Eigenvalues of the first shape
    - evals2: Eigenvalues of the second shape
    """
    def __init__(self, weight=1):
        super().__init__()
        self.weight = weight

    required_inputs = ["Cxy", "Cxy_sup"]
    def forward(self, Cxy, Cxy_sup):
        metric = SquaredFrobeniusLoss()
        return self.weight * metric(Cxy,Cxy_sup)
