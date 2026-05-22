"""Deep functional map matcher via per-pair gradient descent."""

import copy

import torch

from geomfum.learning.losses import (
    BijectivityLoss,
    LossManager,
    OrthonormalityLoss,
)
from geomfum.learning.models import FMNet
from geomfum.matcher.base import BaseMatcher, CorrespondenceResult


class DeepFMMatcher(BaseMatcher):
    """Deep functional map matcher optimized per shape pair.

    Jointly optimizes a deep FM model (feature extractor + functional map solver)
    for a single shape pair using gradient descent, with no dataset-level training.
    The model is deep-copied at the start of each call, so every pair is solved
    independently.  Passing a pretrained model turns this into a per-pair
    fine-tuning step.

    The optimization loop is identical to the learning stage:

        model.forward() → LossManager.compute_loss() → loss.backward()

    This means any ``BaseModel`` (``FMNet``, ``RobustFMNet``, …) and any
    combination of losses from ``geomfum.learning.losses`` can be plugged in.

    Parameters
    ----------
    model : BaseModel, optional
        Deep FM model to optimize. Defaults to ``FMNet()`` (random DiffusionNet
        weights). Pass a pretrained model instance for warm-start fine-tuning.
    loss_manager : LossManager, optional
        Weighted combination of losses. Defaults to orthonormality + bijectivity
        + spectral descriptor preservation (all weight 1).
    fmap_size : int or tuple of int
        Number of LBO eigenfunctions for the functional map.
        A tuple ``(k_b, k_a)`` allows different sizes per shape.
    n_iters : int
        Gradient descent iterations per pair.
    lr : float
        Adam learning rate.
    verbose : bool
        If True, print each loss component every 100 iterations.
    """

    def __init__(
        self,
        model=None,
        loss_manager=None,
        fmap_size=30,
        n_iters=1000,
        lr=1e-3,
        verbose=False,
    ):
        self.model = model or FMNet()
        self.loss_manager = loss_manager or self._default_loss_manager()
        self.fmap_size = (
            fmap_size if isinstance(fmap_size, tuple) else (fmap_size, fmap_size)
        )
        self.n_iters = n_iters
        self.lr = lr
        self.verbose = verbose

    @staticmethod
    def _default_loss_manager():
        return LossManager(
            [
                OrthonormalityLoss(weight=1.0),
                BijectivityLoss(weight=1.0),
            ]
        )

    def __call__(self, shape_a, shape_b):
        """Optimize per-pair and return correspondences.

        Both shapes must have a precomputed spectral basis (``shape.basis``).
        The basis eigenvectors and mass matrix should already be torch tensors
        (same requirement as for FMNet inference).

        Parameters
        ----------
        shape_a : Shape
            First shape (target for p2p21).
        shape_b : Shape
            Second shape (source for p2p21).

        Returns
        -------
        result : CorrespondenceResult
            Contains ``fmap12``, ``fmap21``, ``p2p21``, ``p2p12``,
            ``descr_a``, ``descr_b``.
        """
        k_b, k_a = self.fmap_size
        shape_a.basis.use_k = k_a
        shape_b.basis.use_k = k_b

        device = shape_a.basis.vals.device
        model = copy.deepcopy(self.model).to(device)
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        for i in range(self.n_iters):
            optimizer.zero_grad()

            outputs = model(shape_a, shape_b, as_dict=True)
            outputs["shape_a"] = shape_a
            outputs["shape_b"] = shape_b

            total_loss, loss_dict = self.loss_manager.compute_loss(outputs)
            total_loss.backward()
            optimizer.step()

            if self.verbose and (i % 100 == 0 or i == self.n_iters - 1):
                parts = "  ".join(f"{k}={v:.4f}" for k, v in loss_dict.items())
                print(f"  iter {i:4d} | {parts}")

        model.eval()
        with torch.no_grad():
            outputs = model(shape_a, shape_b, as_dict=True)

        return CorrespondenceResult(
            fmap12=outputs["fmap12"],
            fmap21=outputs.get("fmap21"),
            p2p21=outputs.get("p2p21"),
            p2p12=outputs.get("p2p12"),
            descr_a=outputs.get("desc_a"),
            descr_b=outputs.get("desc_b"),
        )
