"""Losses for Deep Functional Maps training."""

import torch
import torch.nn as nn

import geomfum.linalg as la


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
        self.metric = SquaredFrobeniusLoss()

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
        eye_b = torch.eye(fmap12.shape[1], device=fmap12.device)
        eye_a = torch.eye(fmap21.shape[0], device=fmap21.device)
        return self.weight * (
            self.metric(torch.mm(fmap12.T, fmap12), eye_b)
            + self.metric(torch.mm(fmap21.T, fmap21), eye_a)
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
        self.metric = SquaredFrobeniusLoss()

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
        eye_b = torch.eye(fmap12.shape[0], device=fmap12.device)
        eye_a = torch.eye(fmap21.shape[0], device=fmap21.device)
        return self.weight * self.metric(
            torch.mm(fmap12, fmap21), eye_b
        ) + self.weight * self.metric(torch.mm(fmap21, fmap12), eye_a)


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
        self.metric = SquaredFrobeniusLoss()

    required_inputs = ["fmap12", "fmap21", "shape_a", "shape_b"]

    def forward(self, fmap12, fmap21, shape_a, shape_b):
        """
        Forward pass.

        Parameters
        ----------
        fmap12 : torch.Tensor
            Functional map tensor from source to target shape, of shape ( spectrum_size_b, spectrum_size_a ).
        shape_a : Shape
            Shape object containing source shape information.
        shape_b : Shape
            Shape object containing target shape information.

        Returns
        -------
        torch.Tensor
            Scalar tensor representing the weighted squared Frobenius norm of the Laplacian commutativity error.
        """
        return self.weight * self.metric(
            torch.einsum("bc,c->bc", fmap12, shape_b.basis.vals),
            torch.einsum("b,bc->bc", shape_a.basis.vals, fmap12),
        ) + self.weight * self.metric(
            torch.einsum("bc,c->bc", fmap21, shape_a.basis.vals),
            torch.einsum("b,bc->bc", shape_b.basis.vals, fmap21),
        )


class Fmap_Supervision(nn.Module):
    """
    Computes the supervision loss between predicted and ground truth functional maps.

    Parameters
    ----------
    weight : float, optional
        Weight for the loss term (default: 1).
    """

    def __init__(self, weight=1):
        super().__init__()
        self.weight = weight
        self.metric = SquaredFrobeniusLoss()

    required_inputs = ["fmap12", "fmap12_sup"]

    def forward(self, fmap12, fmap12_sup):
        """
        Forward pass.

        Parameters
        ----------
        fmap12 : torch.Tensor
            Functional map tensor from source to target shape, of shape (batch_size, dim_out, dim_in).
        fmap12_sup : torch.Tensor
            Supervised functional map tensor from source to target shape, of shape (batch_size, dim_out, dim_in).

        Returns
        -------
        torch.Tensor
            Scalar tensor representing the weighted squared Frobenius norm of the difference between predicted and supervised functional maps.
        """
        return self.weight * self.metric(fmap12, fmap12_sup)


class DescriptorCommutativityLoss(nn.Module):
    """
    Computes the descriptor commutativity loss for learning scenarios.

    This loss enforces that functional maps commute with multiplication operators
    derived from descriptors. It's equivalent to OperatorCommutativityEnforcing.from_multiplication
    but designed for PyTorch training.

    Parameters
    ----------
    weight: float, optional
        Weight for the loss term (default: 1).
    """

    def __init__(self, weight=1):
        super().__init__()
        self.weight = weight
        self.metric = SquaredFrobeniusLoss()

    required_inputs = ["fmap12", "fmap21", "desc_a", "desc_b", "shape_a", "shape_b"]

    def _compute_multiplication_operators(self, basis, desc):
        """
        Compute multiplication operators for descriptors.

        Parameters
        ----------
        basis : Basis
            Basis object containing eigenvectors and pseudo-inverse.
        desc : torch.Tensor
            Descriptors of shape (num_vertices, num_descriptors).

        Returns
        -------
        operators : torch.Tensor
            Multiplication operators of shape (num_descriptors, spectrum_size, spectrum_size).
        """
        # desc: (num_vertices, num_descriptors)
        # basis.vecs: (num_vertices, spectrum_size)
        # basis.pinv: (spectrum_size, num_vertices)

        operators = []
        for desc_i in desc:
            operator = basis.pinv @ la.rowwise_scaling(desc_i, basis.vecs)
            operators.append(operator)

        return torch.stack(operators)  # (num_descriptors, spectrum_size, spectrum_size)

    def forward(self, fmap12, fmap21, desc_a, desc_b, shape_a, shape_b):
        """
        Forward pass.

        Parameters
        ----------
        fmap12 : torch.Tensor
            Functional map tensor from shape 1 to shape 2 of shape (spectrum_size_b, spectrum_size_a).
        fmap21 : torch.Tensor
            Functional map tensor from shape 2 to shape 1 of shape (spectrum_size_a, spectrum_size_b).
        desc_a : torch.Tensor
            Descriptors for shape A of shape (num_vertices_a, num_descriptors).
        desc_b : torch.Tensor
            Descriptors for shape B of shape (num_vertices_b, num_descriptors).
        shape_a : TriangleMesh or PointCloud
            TriangleMesh object containing source shape information.
        shape_b : TriangleMesh or PointCloud
            TriangleMesh object containing target shape information.

        Returns
        -------
        torch.Tensor
            Scalar tensor representing the weighted descriptor commutativity loss.
        """
        # Compute multiplication operators for each descriptor
        oper_a = self._compute_multiplication_operators(shape_a.basis, desc_a)
        oper_b = self._compute_multiplication_operators(shape_b.basis, desc_b)

        total_loss = 0
        # Compute commutativity loss for each descriptor
        for oper_a_i, oper_b_i in zip(oper_a, oper_b):
            left_side = torch.mm(fmap12, oper_a_i)  # (spectrum_size_b, spectrum_size_a)
            right_side = torch.mm(
                oper_b_i, fmap12
            )  # (spectrum_size_b, spectrum_size_a)
            loss_12 = self.metric(left_side, right_side)

            # For fmap21: C21 @ M_b = M_a @ C21
            left_side_21 = torch.mm(
                fmap21, oper_b_i
            )  # (spectrum_size_a, spectrum_size_b)
            right_side_21 = torch.mm(
                oper_a_i, fmap21
            )  # (spectrum_size_a, spectrum_size_b)
            loss_21 = self.metric(left_side_21, right_side_21)

            total_loss += loss_12 + loss_21

        total_loss = total_loss / oper_a.shape[0]

        return self.weight * total_loss


class GroundTruthSupervisionLoss(nn.Module):
    """
    Computes the loss of a functional map by measuring the discrepancy between the functional map and a ground truth functional map.

    Parameters
    ----------
    weight : float, optional
        Weight for the loss term (default: 1).
    """

    def __init__(self, weight=1):
        super().__init__()
        self.weight = weight
        self.metric = SquaredFrobeniusLoss()

    required_inputs = ["fmap12", "fmap21", "shape_a", "shape_b", "corr_a", "corr_b"]

    def _compute_ground_truth_map(self, shape_a, shape_b, corr_a, corr_b):
        """Compute the ground truth functional maps.

        Parameters
        ----------
        shape_a : TriangleMesh
            TriangleMesh object containing source shape information.
        shape_b : TriangleMesh
            TriangleMesh object containing target shape information.
        corr_a : torch.Tensor
            Indices of source correspondences.
        corr_b : torch.Tensor
            Indices of target correspondences.

        Returns
        -------
        fmap12_gt ,fmap21_gt : torch.Tensor
            Ground truth functional maps from shape 1 to shape 2 and from shape 2 to shape 1.
        """
        fmap12_gt = shape_b.basis.pinv[:, corr_b] @ shape_a.basis.vecs[corr_a, :]

        fmap21_gt = shape_a.basis.pinv[:, corr_a] @ shape_b.basis.vecs[corr_b, :]

        return fmap12_gt, fmap21_gt

    def forward(self, fmap12, fmap21, shape_a, shape_b, corr_a, corr_b):
        """
        Forward pass.

        Parameters
        ----------
        fmap12 : torch.Tensor
            Functional map tensor from shape 1 to shape 2 of shape (spectrum_size_b, spectrum_size_a).
        fmap21 : torch.Tensor
            Functional map tensor from shape 2 to shape 1 of shape (spectrum_size_a, spectrum_size_b).
        shape_a : TriangleMesh
            TriangleMesh object containing source shape information.
        shape_b : TriangleMesh
            TriangleMesh object containing target shape information.
        corr_a : torch.Tensor
            Indices of source correspondences.
        corr_b : torch.Tensor
            Indices of target correspondences.

        Returns
        -------
        torch.Tensor
            Scalar tensor representing the weighted mean squared Frobenius norm between fmap12 and the ground truth functional map, and between fmap21 and the ground truth functional map.
        """
        fmap12_gt, fmap21_gt = self._compute_ground_truth_map(
            shape_a, shape_b, corr_a, corr_b
        )
        return self.weight * self.metric(fmap12, fmap12_gt) + self.weight * self.metric(
            fmap21, fmap21_gt
        )


class FmapDescriptorsSupervisionLoss(nn.Module):
    """
    Computes the loss of a functional map by measuring the discrepancy between the functional map and a functional map computed by the similarity of the descriptors.

    Parameters
    ----------
    weight : float, optional
        Weight for the loss term (default: 1).
    """

    def __init__(self, weight=1):
        super().__init__()
        self.weight = weight
        self.metric = SquaredFrobeniusLoss()

    required_inputs = ["fmap12", "fmap21", "fmap12_desc", "fmap21_desc"]

    def forward(self, fmap12, fmap21, fmap12_desc, fmap21_desc):
        """
        Forward pass.

        Parameters
        ----------
        fmap12 : torch.Tensor
            Functional map tensor from shape 1 to shape 2 of shape (spectrum_size_b, spectrum_size_a).
        fmap21 : torch.Tensor
            Functional map tensor from shape 2 to shape 1 of shape (spectrum_size_a, spectrum_size_b).
        fmap12_desc : torch.Tensor
            Functional map from the descriptor similarity tensor from shape 1 to shape 2 of shape (spectrum_size_b, spectrum_size_a).
        fmap21_desc : torch.Tensor
            Functional map from the descriptor similarity tensor from shape 2 to shape 1 of shape (spectrum_size_a, spectrum_size_b).

        Returns
        -------
        torch.Tensor
            Scalar tensor representing the weighted mean squared Frobenius norm between fmap12 and fmap12_desc, and between fmap21 and fmap21_desc.
        """
        return self.weight * self.metric(
            fmap12, fmap12_desc
        ) + self.weight * self.metric(fmap21, fmap21_desc)


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
