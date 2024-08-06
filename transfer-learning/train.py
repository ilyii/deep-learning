# train.py

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from config import DEVICE, SAVE_PATH
from model import CNN

def train(
    epochs: int, 
    model: nn.Module, 
    dataloader_train: DataLoader, 
    dataloader_val: DataLoader, 
    optimizer: Optimizer, 
    criterion: nn.Module, 
    val_freq: int = 1
):
    """Train the model with the given dataloaders, optimizer, and criterion."""
    metrics = {
        "train": {"acc": [], "loss": []},
        "val": {"acc": [], "loss": []}
    }

    for epoch in range(epochs):
        model.train()
        with tqdm(dataloader_train, unit="batch", total=len(dataloader_train), desc=f"[TRAIN] Epoch {epoch}/{epochs}") as trainbar:
            for images, targets in trainbar:
                images, targets = images.to(DEVICE), targets.to(DEVICE)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, targets)

                predicted = torch.argmax(outputs, 1)
                correct = (predicted == targets).sum().item()

                metrics["train"]["acc"].append(correct / len(targets))
                metrics["train"]["loss"].append(loss.item())

                loss.backward()
                optimizer.step()

                trainbar.set_postfix(loss=loss.item())

        if (epoch + 1) % val_freq == 0:
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                with tqdm(dataloader_val, unit="batch", total=len(dataloader_val), desc=f"[VAL] Epoch {epoch}/{epochs}") as valbar:
                    for images, targets in valbar:
                        images, targets = images.to(DEVICE), targets.to(DEVICE)

                        outputs = model(images)
                        loss = criterion(outputs, targets)

                        _, predicted = torch.max(outputs.data, 1)
                        total += targets.size(0)
                        correct += (predicted == targets).sum().item()

                        metrics["val"]["acc"].append(correct / total)
                        metrics["val"]["loss"].append(loss.item())

                        valbar.set_postfix(acc=correct / total)

    return metrics

def save_model(model: nn.Module, filename: str):
    """Save the model state dictionary."""
    os.makedirs(SAVE_PATH, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(SAVE_PATH, filename))
