"""Trainer for Deep Functional Maps (DFM) using PyTorch."""

import logging

import torch
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
import random


# helper function
def get_dataset_attr(dataset, attr):
    # Recursively get the attribute from Subset or the base dataset
    while isinstance(dataset, torch.utils.data.Subset):
        dataset = dataset.dataset
    return getattr(dataset, attr)


class DeepFunctionalMapTrainer:
    """Trainer for Deep Functional Maps (DFM) using PyTorch.

    Parameters
    ----------
    model : nn.Module
        The model to be trained, typically a subclass of nn.Module.
    Loss : LossManager
        Loss manager that computes the loss during training.
    train_set : PairDataset
        Dataset for the training.
    val_set : PairDataset
        Dataset for the validation.
    optimizer : torch.optim.Optimizer
        Optimizer for updating model parameters.
    scheduler : torch.optim.lr_scheduler, optional
        Learning rate scheduler for adjusting the learning rate during training
        (default is None).
    scheduler_step_on : str, optional
        When to step the scheduler: "epoch" or "batch" (default is "epoch").
    epochs : int, optional
        Number of epochs to train the model (default is 100).
    device : str, optional
        Device to run the training on, either "cuda" or "cpu" (default is "cuda").
    checkpoint_path : str, optional
        Path to save the model checkpoints (default is None, no checkpointing).
    monitor_metric : str, optional
        Metric to monitor for saving checkpoints (default is "loss").
    mode : str, optional
        Mode for monitoring the metric, either "min" or "max" (default is "min").
    """

    def __init__(
        self,
        model,
        train_set,
        val_set,
        train_loss_manager=None,
        val_loss_manager=None,
        optimizer=None,
        scheduler=None,
        scheduler_step_on="epoch",
        epochs=100,
        device="cpu",
        checkpoint_path=None,
        monitor_metric="loss",
        mode="min",
    ):
        self.model = model
        self.train_loss_manager = train_loss_manager
        self.val_loss_manager = val_loss_manager

        if self.val_loss_manager is None:
            self.val_loss_manager = train_loss_manager

        self.train_set = train_set
        self.val_set = val_set
        self.optimizer = optimizer
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=1e-3, weight_decay=1e-5
            )
        self.scheduler = scheduler
        self.scheduler_step_on = scheduler_step_on
        self.epochs = epochs
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.monitor_metric = monitor_metric
        self.mode = mode

    def train_one_epoch(self, epoch):
        """Train the model for one epoch."""
        self.model.train()
        running_loss = 0.0
        indices = list(range(len(self.train_set)))
        random.shuffle(indices)  # Shuffle indices

        with tqdm(
            total=len(self.train_set),
            desc=f"Epoch {epoch + 1}/{self.epochs} (Train)",
            unit="batch",
        ) as pbar:
            for idx in indices:
                pair = self.train_set[idx]  # Access item by index
                self.optimizer.zero_grad()
                shape_a = pair["source"]["shape"]
                shape_b = pair["target"]["shape"]
                outputs = self.model(shape_a, shape_b)
                outputs.update(
                    {
                        "shape_a": shape_a,
                        "shape_b": shape_b,
                    }
                )
                if get_dataset_attr(self.train_set.shape_data, "correspondences"):
                    outputs.update(
                        {
                            "corr_a": pair["source"]["corr"],
                            "corr_b": pair["target"]["corr"],
                        }
                    )
                if get_dataset_attr(self.train_set.shape_data, "distances"):
                    outputs.update(
                        {
                            "dist_a": pair["source"]["dist_matrix"],
                            "dist_b": pair["target"]["dist_matrix"],
                        }
                    )

                loss, loss_dict = self.train_loss_manager.compute_loss(outputs)

                loss.backward()
                self.optimizer.step()

                # Step scheduler if configured to step on batch
                if self.scheduler and self.scheduler_step_on == "batch":
                    self.scheduler.step()

                running_loss += loss.item()
                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
                pbar.update(1)
        avg_train_loss = running_loss / len(self.train_set)
        return avg_train_loss

    def validate(self, return_metrics=False):
        """Validate the model on the validation dataset."""
        self.model.eval()
        val_loss = 0.0
        metrics_sum = {}
        with torch.no_grad():
            with tqdm(total=len(self.val_set), desc="Validation", unit="batch") as pbar:
                for pair in self.val_set:
                    shape_a = pair["source"]["shape"]
                    shape_b = pair["target"]["shape"]
                    outputs = self.model(shape_a, shape_b)
                    outputs.update(
                        {
                            "shape_a": shape_a,
                            "shape_b": shape_b,
                        }
                    )
                    if get_dataset_attr(self.val_set.shape_data, "correspondences"):
                        outputs.update(
                            {
                                "corr_a": pair["source"]["corr"],
                                "corr_b": pair["target"]["corr"],
                            }
                        )
                    if get_dataset_attr(self.val_set.shape_data, "distances"):
                        outputs.update(
                            {
                                "dist_a": pair["source"]["dist_matrix"],
                                "dist_b": pair["target"]["dist_matrix"],
                            }
                        )
                    loss, loss_dict = self.val_loss_manager.compute_loss(outputs)
                    val_loss += loss.item()
                    for k, v in loss_dict.items():
                        metrics_sum[k] = metrics_sum.get(k, 0.0) + v
                    pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
                    pbar.update(1)
        avg_val_loss = val_loss / len(self.val_set)
        avg_metrics = {k: v / len(self.val_set) for k, v in metrics_sum.items()}
        if return_metrics:
            return avg_val_loss, avg_metrics
        return avg_val_loss

    def train(self):
        """Train the model for the specified number of epochs."""
        best_metric = float("inf") if self.mode == "min" else -float("inf")
        for epoch in range(self.epochs):
            logging.info(f"Epoch [{epoch + 1}/{self.epochs}] - Training")
            avg_train_loss = self.train_one_epoch(epoch)
            logging.info(
                f"Epoch [{epoch + 1}/{self.epochs}] - Average Training Loss: {avg_train_loss:.4f}"
            )

            avg_val_loss, val_metrics = self.validate(return_metrics=True)
            logging.info(
                f"Epoch [{epoch + 1}/{self.epochs}] - Average Validation Loss: {avg_val_loss:.4f}"
            )

            # Step scheduler if configured to step on epoch
            if self.scheduler and self.scheduler_step_on == "epoch":
                # Handle different types of schedulers
                if isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    # ReduceLROnPlateau needs a metric to monitor
                    metric_value = val_metrics.get(self.monitor_metric, avg_val_loss)
                    self.scheduler.step(metric_value)
                else:
                    # Other schedulers don't need arguments
                    self.scheduler.step()

            metric_value = val_metrics.get(self.monitor_metric, avg_val_loss)
            improved = (
                metric_value < best_metric
                if self.mode == "min"
                else metric_value > best_metric
            )
            if self.checkpoint_path and improved:
                best_metric = metric_value
                checkpoint_dict = {
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "epoch": epoch,
                    "best_metric": best_metric,
                }
                # Save scheduler state if scheduler exists
                if self.scheduler:
                    checkpoint_dict["scheduler_state_dict"] = (
                        self.scheduler.state_dict()
                    )

                torch.save(checkpoint_dict, self.checkpoint_path)
                logging.info(
                    f"Checkpoint saved at epoch {epoch + 1} with {self.monitor_metric}: {metric_value:.4f}"
                )

    def load_checkpoint(self, checkpoint_path):
        """Load model, optimizer, and scheduler states from a checkpoint.

        Parameters
        ----------
        checkpoint_path : str
            Path to the checkpoint file to load.

        Returns
        -------
        dict
            Dictionary containing epoch and best_metric information.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Load scheduler state if it exists and scheduler is configured
        if self.scheduler and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        logging.info(f"Checkpoint loaded from {checkpoint_path}")
        logging.info(f"Resuming from epoch {checkpoint['epoch'] + 1}")

        return {"epoch": checkpoint["epoch"], "best_metric": checkpoint["best_metric"]}
