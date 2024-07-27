import torch
import torchvision
from torch.utils.data import DataLoader

import config


TRANSFORM = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((config.MEAN,), (config.STD,))
    ])

def load_data(datapath, batchsize):
    trainset = torchvision.datasets.MNIST(datapath, train=True, download=True, transform=TRANSFORM)
    trainloader = DataLoader(trainset, batch_size=batchsize, shuffle=True)

    testset = torchvision.datasets.MNIST(datapath, train=False, download=True, transform=TRANSFORM)
    testloader = DataLoader(testset, batch_size=batchsize, shuffle=False)

    return trainloader, testloader