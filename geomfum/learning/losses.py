"""Losses for functional maps."""

import torch
import torch.nn as nn


class LossManager:
    """
    Manages a list of loss functions and their weights for model training.

    Parameters
    ----------
    losses : list of (nn.Module, float) or list of nn.Module
        List of (loss_module, weight) tuples, or just loss modules (weight=1.0).
    """

    def __init__(self, losses):
        self.losses = losses

    def compute_loss(self, outputs):
        """Compute the total loss and a dictionary of individual losses.

        Parameters
        ----------
        outputs : dict
            Dictionary containing the outputs of the model, which should include all required inputs for the loss functions

        Returns
        -------
        total_loss : torch.Tensor
            Scalar tensor representing the total loss computed from all loss functions.
        loss_dict : dict
            Dictionary mapping loss function names to their computed values.
        """
        total_loss = 0
        loss_dict = {}
        for loss_fn in self.losses:
            # Get required input keys for this loss
            required_keys = getattr(loss_fn, "required_inputs", None)
            if required_keys is not None:
                args = [outputs[k] for k in required_keys]
                loss_value = loss_fn(*args)
            else:
                # fallback: pass the whole dict
                loss_value = loss_fn(outputs)
            name = loss_fn.__class__.__name__
            loss_dict[name] = loss_value.item()
            total_loss += loss_value
        return total_loss, loss_dict


######################LOSS IMPLEMENTATIONS ############################


class SquaredFrobeniusLoss(nn.Module):
    """
    Computes the mean squared Frobenius norm between two input tensors.

    Parameters
    ----------
    None
    """

    def forward(self, a, b):
        """
        Forward pass.

        Parameters
        ----------
        a : torch.Tensor
            First input tensor matrix.
        b : torch.Tensor
            Second input tansor matrix, must be broadcastable to the shape of `a`.

        Returns
        -------
        torch.Tensor
            Scalar tensor representing the mean squared Frobenius norm between `a` and `b`.
        """
        return torch.mean(torch.sum(torch.abs(a - b) ** 2, dim=(-2, -1)))


class OrthonormalityLoss(nn.Module):
    """
    Computes the orthonormality error of a functional map by measuring the mean squared Frobenius norm between C^T C and the identity matrix.

    Parameters
    ----------
    weight : float, optional
        Weight for the loss term (default: 1).
    """

    def __init__(self, weight=1):
        super().__init__()
        self.weight = weight

    required_inputs = ["fmap12", "fmap21"]

    def forward(self, fmap12, fmap21):
        """
        Forward pass.

        Parameters
        ----------
        fmap12 : torch.Tensor
            Functional map tensor of shape ( spectrum_size_b, spectrum_size_a).
        fmap21 : torch.Tensor
            Functional map tensor of shape ( spectrum_size_a, spectrum_size_b).

        Returns
        -------
        torch.Tensor
            Scalar tensor representing the weighted mean squared Frobenius norm between C^T C and the identity matrix.
        """
        metric = SquaredFrobeniusLoss()
        eye_b = torch.eye(fmap12.shape[0], device=fmap12.device)
        eye_a = torch.eye(fmap21.shape[1], device=fmap21.device)
        return self.weight * (
            metric(torch.mm(fmap12.T, fmap12), eye_b)
            + metric(torch.mm(fmap21.T, fmap21), eye_a)
        )


class BijectivityLoss(nn.Module):
    """
    Computes the bijectivity error of two functional maps by measuring the mean squared Frobenius norm between fmap12 fmap21 and the identity matrix, and between fmap21 fmap12 and the identity matrix.

    Parameters
    ----------
    weight : float, optional
        Weight for the loss term (default: 1).
    """

    def __init__(self, weight=1):
        super().__init__()
        self.weight = weight

    required_inputs = ["fmap12", "fmap21"]

    def forward(self, fmap12, fmap21):
        """
        Forward pass.

        Parameters
        ----------
        fmap12 : torch.Tensor
            Functional map tensor from shape 1 to shape 2 of shape (spectrum_size_b, spectrum_size_a).
        fmap21 : torch.Tensor
            Functional map tensor from shape 2 to shape 1 of shape (spectrum_size_a, spectrum_size_b).

        Returns
        -------
        torch.Tensor
            Scalar tensor representing the weighted mean squared Frobenius norm between fmap12 fmap21 and the identity matrix, and between fmap21 fmap12 and the identity matrix.
        """
        metric = SquaredFrobeniusLoss()
        eye_b = torch.eye(fmap12.shape[0], device=fmap12.device)
        eye_a = torch.eye(fmap21.shape[1], device=fmap21.device)
        return self.weight * metric(torch.mm(fmap12, fmap21), eye_b) + self.weight * metric(
            torch.mm(fmap21, fmap12), eye_a
        )


class LaplacianCommutativityLoss(nn.Module):
    """
    Computes the Laplacian commutativity error of a functional map by measuring the discrepancy between the action of the Laplacian eigenvalues and the functional map.

    Parameters
    ----------
    weight : float, optional
        Weight for the loss term (default: 1).
    """

    def __init__(self, weight=1):
        super().__init__()
        self.weight = weight

    required_inputs = ["fmap12", "mesh_a", "mesh_b"]

    def forward(self, fmap12, mesh_a, mesh_b):
        """
        Forward pass.

        Parameters
        ----------
        fmap12 : torch.Tensor
            Functional map tensor from source to target shape, of shape ( spectrum_size_b, spectrum_size_a ).
        mesh_a : TriangleMesh
            TriangleMesh object containing source shape information.
        mesh_b : TriangleMesh
            TriangleMesh object containing target shape information.

        Returns
        -------
        torch.Tensor
            Scalar tensor representing the weighted squared Frobenius norm of the Laplacian commutativity error.
        """
        metric = SquaredFrobeniusLoss()
        return self.weight * metric(
            torch.einsum("bc,c->bc", fmap12, mesh_b.basis.vals),
            torch.einsum("b,bc->bc", mesh_a.basis.vals, fmap12),
        )

class GeodesicError(nn.Module):
    """
    Computes the accuracy of a correspondence by measuring the mean of the geodesic distances between points of the predicted permuted target and the ground truth target.

    Parameters
    ----------
    None
    """

    def __init__(self):
        super().__init__()

    required_inputs = [
        "p2p12",
        "dist_b",
        "corr_a",
        "corr_b",
    ]

    def _compute_geodesic_loss(self, p2p, target_dist, source_corr, target_corr):
        """
        Compute the geodesic loss for batched inputs.

        Parameters
        ----------
        p2p : torch.Tensor
            Predicted point-to-point map.
        target_dist : torch.Tensor
            Geodesic distance matrix for the target shape.
        source_corr : torch.Tensor
            Indices of source correspondences.
        target_corr : torch.Tensor
            Indices of target correspondences.

        Returns
        -------
        torch.Tensor
            Mean geodesic distance error.
        """
        return torch.mean(target_dist[p2p[source_corr], target_corr])

    def forward(self, p2p12, dist_b, corr_a, corr_b):
        """
        Forward pass.

        Parameters
        ----------
        p2p12 : torch.Tensor
            Predicted point-to-point map.
        dist_b : torch.Tensor
            Geodesic distance matrix for the target shape.
        corr_a : torch.Tensor
            Indices of source correspondences.
        corr_b : torch.Tensor
            Indices of target correspondences.

        Returns
        -------
        torch.Tensor
            Mean geodesic distance error.
        """
        loss = self._compute_geodesic_loss(p2p12, dist_b, corr_a, corr_b)
        return loss
