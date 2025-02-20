import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from geomfum.dfm.forward_functional_map import ForwardFunctionalMap
from geomfum.descriptor.learned import LearnedDescriptor

from geomfum.dfm.losses import LossManager
from geomfum.dfm.dataset import ShapeDataset

class DeepFunctionalMapTrainer:
    def __init__(self, config):
        """
        Initializes the trainer with a configuration dictionary.
        
        Args:
            config (dict): Dictionary containing all configurations.
        """
        self.config = config
        self.device = config.get("device", "cuda")
        self.epochs = config.get("epochs", 100)
        self.batch_size = config.get("batch_size", 1)
        self.lr = config.get("lr", 0.001)
        self.shape_dir = config["shape_dir"]

        # Dataset & DataLoader
        self.train_loader, self.test_loader = self._get_dataloaders()

        # Model, Loss, Optimizer
        self.descr, self.forward_map, self.loss_manager, self.optimizer = self._initialize_training_components()

    def _get_dataloaders(self):
        dataset = ShapeDataset(self.shape_dir, pair_mode=self.config.get("pair_mode", "all"), device=self.device)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        return train_loader, test_loader

    def _initialize_training_components(self):
        descr = LearnedDescriptor.from_registry(
            which=self.config.get("descriptor", "diffusion_net"), 
            cache_dir=self.shape_dir + "diffusion/", 
            device=self.device
        )
        forward_map = ForwardFunctionalMap(
            self.config.get("functional_map_lambda", 0.01), 
            self.config.get("functional_map_gamma", 1)
        )

        loss_config = self.config.get("loss_config", {"Orthonormality": 1.0, "Laplacian_Commutativity": 0.01})
        loss_manager = LossManager(loss_config)

        optimizer = torch.optim.Adam(descr.model.parameters(), lr=self.lr)
        return descr, forward_map, loss_manager, optimizer

    def train(self):
        print(f"Training for {self.epochs} epochs...")

        for epoch in range(self.epochs):
            print(f"Epoch {epoch + 1}/{self.epochs}")
            running_loss = 0.0

            for batch_idx, (source, target) in enumerate(tqdm(self.train_loader)):
                self.optimizer.zero_grad()

                # Compute features
                feat_a = self.descr(source)
                feat_b = self.descr(target)

                # Compute functional maps
                Cxy = self.forward_map(source, target, feat_a, feat_b)

                # Compute loss
                loss, loss_details = self.loss_manager.compute_loss(
                    Cxy=Cxy, evals_x=source["evals"], evals_y=target["evals"]
                )

                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                if (batch_idx + 1) % 10 == 0:
                    print(f"Step {batch_idx + 1}/{len(self.train_loader)} - Loss: {loss.item():.4f}, Breakdown: {loss_details}")

            avg_loss = running_loss / len(self.train_loader)
            print(f"Epoch {epoch + 1}/{self.epochs}, Average Loss: {avg_loss:.4f}")

            # Save model checkpoint
            if (epoch + 1) % self.config.get("save_interval", 50) == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pth")

    def test(self):
        print("Testing...")
        test_loss = 0.0

        with torch.no_grad():
            for batch_idx, (source, target) in enumerate(tqdm(self.test_loader)):
                feat_a = self.descr(source)
                feat_b = self.descr(target)

                Cxy = self.forward_map(source, target, feat_a, feat_b)
                loss, loss_details = self.loss_manager.compute_loss(
                    Cxy=Cxy, evals_x=source["evals"], evals_y=target["evals"]
                )

                test_loss += loss.item()

                if (batch_idx + 1) % 10 == 0:
                    print(f"Step {batch_idx + 1}/{len(self.test_loader)} - Loss: {loss.item():.4f}, Breakdown: {loss_details}")

        avg_test_loss = test_loss / len(self.test_loader)
        print(f"Average Test Loss: {avg_test_loss:.4f}")

    def save_checkpoint(self, path):
        torch.save(self.descr.model.state_dict(), path)
        print(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path):
        self.descr.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Checkpoint loaded: {path}")
