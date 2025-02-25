"""
This file contains the trainer for the Deep functional map model.
This code is based on the assumption that this file should not be modified bu user or developer to test different models and so it is just a way to instantiate
models and losses defined in their original code.
"""


import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import logging
from geomfum.dfm.losses import LossManager
from geomfum.dfm.dataset import PairsDataset
from geomfum.dfm.model import get_model_class

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DeepFunctionalMapTrainer:
    def __init__(self, config):
        self.config = config
        self.device = config.get("device", "cuda")
        self.epochs = config.get("epochs", 100)
        self.batch_size = config.get("batch_size", 1)
        self.lr = config.get("lr", 0.001)
        self.shape_dir = config["shape_dir"]

        # Dataset & DataLoader
        self.train_loader, self.test_loader = self._get_dataloaders()

        # Model, Loss, Optimizer
        self._initialize_training_components()

    def _get_dataloaders(self):
        logging.info("Loading dataset...")
        dataset = PairsDataset(self.shape_dir, pair_mode=self.config.get("pair_mode", "all"), device=self.device)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        logging.info("Dataset loaded successfully.")
        return train_loader, test_loader

    def _initialize_training_components(self):
        logging.info("Initializing model, loss manager, and optimizer...")
        model_class = get_model_class(self.config['model']['class'])
        self.model = model_class(self.config['model']['params'], device=self.device).to(self.device)

        loss_config = self.config.get("loss_config", {"Orthonormality": 1.0, "Laplacian_Commutativity": 0.01})
        self.loss_manager = LossManager(loss_config)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        logging.info("Model, loss manager, and optimizer initialized successfully.")

    def train(self):
        self.model.train()
        for epoch in range(self.epochs):
            running_loss = 0.0
            logging.info(f"Epoch [{epoch+1}/{self.epochs}]")
            with tqdm(total=len(self.train_loader), desc=f"Epoch {epoch+1}/{self.epochs}", unit="batch") as pbar:
                for batch_idx, pair in enumerate(self.train_loader):
                    self.optimizer.zero_grad()
                    outputs = self.model(pair['source'], pair['target'])
                    outputs.update({"source": pair['source'], "target": pair['target']})  # Add source and target to outputs
                    loss, loss_dict = self.loss_manager.compute_loss(outputs)
                    loss.backward()
                    self.optimizer.step()
                    running_loss += loss.item()
                    pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
                    pbar.update(1)
            logging.info(f'Epoch [{epoch+1}/{self.epochs}], Average Loss: {running_loss/len(self.train_loader):.4f}')

    def test(self):
        self.model.eval()
        test_loss = 0.0
        logging.info("Testing...")
        with torch.no_grad():
            with tqdm(total=len(self.test_loader), desc="Testing", unit="batch") as pbar:
                for batch_idx, pair in enumerate(self.test_loader):
                    outputs = self.model(pair['source'], pair['target'])
                    loss, loss_dict = self.loss_manager.compute_loss(outputs)
                    test_loss += loss.item()
                    pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
                    pbar.update(1)
        logging.info(f'Average Test Loss: {test_loss/len(self.test_loader):.4f}')

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        logging.info(f"Model saved to {path}")