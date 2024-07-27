import os
import random
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import torch

import config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def unnorm(img, mean, std):
    return (img.squeeze() * std) + mean


def plot_examples(dataset):
    plt.figure(figsize=(15, 5))
    for i in range(n:=9):
        idx = random.randint(0, len(dataset))
        plt.subplot(1,n,i+1)
        img = unnorm(dataset[idx][0], config.MEAN, config.STD)
        plt.imshow(img, cmap='gray')
        plt.title(f"{dataset[idx][1]}")
        plt.tight_layout()
        plt.axis('off')
    plt.show()


def plot_stats(stats, savepath=None):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    sns.pointplot(x="epoch", y="loss", data=stats, label="Train Loss")   
    sns.pointplot(x="val_epoch", y="val_loss", data=stats, label="Test Loss") if "val_epoch" in stats else None
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    sns.pointplot(x="epoch", y="acc", data=stats, label="Train Acc")
    sns.pointplot(x="val_epoch", y="val_acc", data=stats, label="Test Acc") if "val_epoch" in stats else None
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(savepath, "stats.png")) if savepath else plt.show()