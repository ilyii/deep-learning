import torch
import torchvision
from torchvision import transforms


TRANSFORM = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])


def load_data(root:str, transform:transforms.Compose, batchsize:int):
    trainset = torchvision.datasets.CIFAR10(root=root, train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize,
                                            shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(root=root, train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batchsize,
                                            shuffle=False, num_workers=0)
    return trainloader, testloader

