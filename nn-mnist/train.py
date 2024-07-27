import os
from collections import defaultdict
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd

from data import TRANSFORM, load_data
from model import NN
import config
import util

def train(model: nn.Module, 
          dataloader_train: DataLoader, 
          optimizer: optim.Optimizer, 
          criterion: nn.Module, 
          epochs: int, 
          device: str = "cpu", 
          ckpt_freq: int = 0, 
          savepath: str = "./output") -> dict:
    """
    Train the model.

    Args:
        model (nn.Module): Model to train.
        dataloader_train (DataLoader): Training data loader.
        optimizer (optim.Optimizer): Optimizer.
        criterion (nn.Module): Loss function.
        epochs (int): Number of epochs.
        device (str): Device to train on.
        ckpt_freq (int): Checkpoint frequency.
        savepath (str): Path to save the model checkpoint.

    Returns:
        dict: Training statistics (loss and accuracy per epoch).
    """
    os.makedirs(savepath, exist_ok=True)
    stats = defaultdict(list)

    model.to(device)
    for epoch in range(epochs):
        model.train()
        train_bar = tqdm(dataloader_train, desc=f"[TRAIN] Epoch {epoch+1}/{epochs}")

        epoch_loss = 0.0
        epoch_acc = 0.0

        for img, label in train_bar:
            img, label = img.to(device), label.to(device)

            optimizer.zero_grad()
            output = model(img)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = torch.max(output, 1)
            epoch_acc += (predicted == label).sum().item()

        epoch_loss /= len(dataloader_train)
        epoch_acc /= len(dataloader_train.dataset)

        stats["epoch"].append(epoch)
        stats["loss"].append(epoch_loss)
        stats["acc"].append(epoch_acc)

        train_bar.set_postfix_str(f"Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        # Save checkpoint
        if stats["loss"][-1] == min(stats["loss"]):
            torch.save(model.state_dict(), config.CKPTPATH)

    return stats

if __name__ == "__main__":
    trainloader, _ = load_data(config.DATAPATH, config.BATCHSIZE)
    model = NN(config.IN_DIM, config.HIDDEN_DIMS, config.OUT_DIM)
    optimizer = optim.SGD(model.parameters(), lr=config.LR)
    criterion = nn.CrossEntropyLoss()

    stats = train(model, trainloader, optimizer, criterion, config.NUM_EPOCHS, config.DEVICE)
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(os.path.join(config.SAVEPATH, "stats.csv"), index=False)
    util.plot_stats(stats_df, config.SAVEPATH)