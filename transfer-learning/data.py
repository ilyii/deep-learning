# data.py

import torch
import torchvision
from torchvision.datasets import Flowers102
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor

from config import IMG_SIZE, BATCH_SIZE

def get_datasets():
    dataset_train = Flowers102(root='data/', split="train", download=True)
    dataset_val = Flowers102(root='data/', split="val", download=True)
    dataset_test = Flowers102(root='data/', split="test", download=True)

    print("Train:", len(dataset_train))
    print("Val:", len(dataset_val))
    print("Test:", len(dataset_test))

    return dataset_train, dataset_val, dataset_test

def get_transforms():
    return Compose([
        Resize((IMG_SIZE, IMG_SIZE)),
        ToTensor(),
    ])

def collate_fn(batch):
    images, targets = zip(*batch)
    transform = get_transforms()
    images = torch.stack([transform(image) for image in images])
    targets = torch.tensor(targets)
    return images, targets

def get_dataloaders(dataset_train, dataset_val, dataset_test):
    dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    dataloader_val = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    dataloader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    return dataloader_train, dataloader_val, dataloader_test
