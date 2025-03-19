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
from geomfum.dfm.dataset import ShapeDataset, PairsDataset
from geomfum.dfm.model import get_model_class

from geomfum.convert import P2pFromFmConverter

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
        self.train_loader, self.val_loader = self._get_dataloaders()

        # Model, Loss, Optimizer
        self._initialize_training_components()

    def _get_dataloaders(self):
        logging.info("Loading dataset...")
        dataset = ShapeDataset(self.shape_dir, spectral=True, device=self.device)  # we load all the shapes
        train_size = int(0.8 * len(dataset))        # we split the shapes into train and validation     
        val_size = len(dataset) - train_size       

        train_shapes, validation_shapes = random_split(dataset, [train_size, val_size])
        #we create a dataset of pairs from the training shapes
        train_dataset = PairsDataset(train_shapes, pair_mode=self.config.get("pair_mode", "all"),n_pairs=self.config.get("n_pairs", 100), device=self.device)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        #we create a dataset of pairs from the test shapes
        validation_dataset = PairsDataset(validation_shapes, pair_mode=self.config.get("pair_mode", "all"),n_pairs=self.config.get("n_pairs", 100), device=self.device)
        val_loader = DataLoader(validation_dataset, batch_size=self.batch_size, shuffle=True)

        logging.info("Dataset loaded successfully.")

        return train_loader, val_loader

    def _initialize_training_components(self):
        logging.info("Initializing model, loss manager, and optimizer...")
        model_class = get_model_class(self.config['model']['class'])
        self.model = model_class(self.config['model']['params'], device=self.device).to(self.device)

        loss_config = self.config.get("loss_config", {"Orthonormality": 1.0, "Laplacian_Commutativity": 0.01})
        self.loss_manager = LossManager(loss_config)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        logging.info("Model, loss manager, and optimizer initialized successfully.")


    def train(self):
        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            running_loss = 0.0
            logging.info(f"Epoch [{epoch+1}/{self.epochs}] - Training")
            with tqdm(total=len(self.train_loader), desc=f"Epoch {epoch+1}/{self.epochs} (Train)", unit="batch") as pbar:
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
            avg_train_loss = running_loss / len(self.train_loader)
            logging.info(f"Epoch [{epoch+1}/{self.epochs}] - Average Training Loss: {avg_train_loss:.4f}")

            # Validation phase
            avg_val_loss = self.validate()
            logging.info(f"Epoch [{epoch+1}/{self.epochs}] - Average Validation Loss: {avg_val_loss:.4f}")

    def validate(self):
        self.model.eval()
        val_loss = 0.0
        logging.info("Validating...")
        with torch.no_grad():
            with tqdm(total=len(self.val_loader), desc="Validation", unit="batch") as pbar:
                for batch_idx, pair in enumerate(self.val_loader):
                    outputs = self.model(pair['source'], pair['target'])
                    loss, loss_dict = self.loss_manager.compute_loss(outputs)
                    val_loss += loss.item()
                    pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
                    pbar.update(1)
        avg_val_loss = val_loss / len(self.val_loader)
        logging.info(f"Average Validation Loss: {avg_val_loss:.4f}")
        return avg_val_loss  
    
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        logging.info(f"Model saved to {path}")
    
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        logging.info(f"Model loaded from {path}")