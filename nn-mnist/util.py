import os
import random
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

import torch
import torchvision
from torch.utils.data import DataLoader


class DotDict(dict):
    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = DotDict(value)

    def __getattr__(self, attr):
        if attr in self:
            return self[attr]
        raise AttributeError(f"'DotDict' object has no attribute '{attr}'")

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, attr):
        if attr in self:
            del self[attr]
        else:
            raise AttributeError(f"'DotDict' object has no attribute '{attr}'")

    def __getitem__(self, key):
        return self.get(key, None)

    def __setitem__(self, key, value):
        super(DotDict, self).__setitem__(key, value)

    def __delitem__(self, key):
        super(DotDict, self).__delitem__(key)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_transform(mean, std):
    return torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((mean,), (std,))
    ])


def get_data(opt, split:str, transform):
    dataset = torchvision.datasets.MNIST(opt.datapath, train=(split=="train"), download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size if split=="train" else 1, shuffle=(split=="train"))
    return dataset, dataloader


def unnorm(img, mean, std):
    return (img.squeeze() * std) + mean


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