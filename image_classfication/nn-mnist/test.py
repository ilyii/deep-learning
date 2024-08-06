from typing import Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import TRANSFORM, load_data
from model import NN
import config
import util

def test(model: nn.Module, dataloader_test: DataLoader, criterion: nn.Module, device: str = "cpu") -> Tuple[float, float]:
    """
    Evaluate the model.

    Args:
        model (nn.Module): Model to evaluate.
        dataloader_test (DataLoader): Test data loader.
        criterion (nn.Module): Loss function.
        device (str): Device to test on.

    Returns:
        Tuple[float, float]: Average loss and accuracy.
    """
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_samples = 0

    with torch.no_grad():
        val_bar = tqdm(dataloader_test, desc="[VAL]")

        for img, label in val_bar:
            img, label = img.to(device), label.to(device)

            output = model(img)
            loss = criterion(output, label)

            _, predicted = torch.max(output, 1)
            acc = (predicted == label).sum().item()

            total_loss += loss.item() * img.size(0)
            total_acc += acc
            total_samples += img.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_acc / total_samples

    print(f"Test Loss: {avg_loss:.4f}, Test Acc: {avg_acc:.4f}")


if __name__ == "__main__":
    _, dataloader_test = load_data(datapath=config.DATAPATH, batchsize=config.BATCHSIZE)
    model = NN(config.IN_DIM, config.HIDDEN_DIMS, config.OUT_DIM).to(config.DEVICE)
    model.load_state_dict(torch.load(config.CKPTPATH))
    criterion = nn.CrossEntropyLoss()
    test(model, dataloader_test, criterion, config.DEVICE)
    

