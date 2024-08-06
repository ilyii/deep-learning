# test.py

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from config import DEVICE, SAVE_PATH

def test(model: torch.nn.Module, dataloader_test: DataLoader):
    """Evaluate the model on the test set."""
    metrics = {"correct": 0, "total": 0, "loss": 0}
    model.eval()
    with torch.no_grad():
        with tqdm(dataloader_test, unit="batch", total=len(dataloader_test), desc=f"[TEST]") as testbar:
            for images, targets in testbar:
                images, targets = images.to(DEVICE), targets.to(DEVICE)

                outputs = model(images)
                loss = torch.nn.CrossEntropyLoss()(outputs, targets)
                _, predicted = torch.max(outputs.data, 1)

                metrics["loss"] = loss.item()
                metrics["total"] += targets.size(0)
                metrics["correct"] += (predicted == targets).sum().item()

                testbar.set_postfix(acc=metrics["correct"] / metrics["total"])

    torch.save(model.state_dict(), os.path.join(SAVE_PATH, f"ckpt_{DEVICE}.pt"))

    return metrics
