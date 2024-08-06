# main.py

import torch
import torch.optim as optim
from torchvision import models
import copy
import gc

from config import DEVICE, LR, EPOCHS, VAL_FREQ
from data import get_datasets, get_dataloaders
from model import CNN
from train import train, save_model
from test import test

def main():
    dataset_train, dataset_val, dataset_test = get_datasets()
    dataloader_train, dataloader_val, dataloader_test = get_dataloaders(dataset_train, dataset_val, dataset_test)

    model_own = CNN().to(DEVICE)
    optimizer = optim.Adam(model_own.parameters(), lr=LR)
    criterion = torch.nn.CrossEntropyLoss()

    train_metrics_own = train(EPOCHS, model_own, dataloader_train, dataloader_val, optimizer, criterion, VAL_FREQ)
    test_metrics_own = test(model_own, dataloader_test)
    save_model(model_own, "cnn_own.pt")

    resnet18_frozen = models.resnet18(weights="DEFAULT")
    resnet18_finetune = copy.deepcopy(resnet18_frozen)
    for name, param in resnet18_frozen.named_parameters():
        if "fc" not in name:
            param.requires_grad = False
    resnet18_frozen = resnet18_frozen.to(DEVICE)
    resnet18_finetune = resnet18_finetune.to(DEVICE)

    optimizer = optim.Adam(resnet18_frozen.parameters(), lr=LR)
    train_metrics_res18_fe = train(EPOCHS, resnet18_frozen, dataloader_train, dataloader_val, optimizer, criterion, VAL_FREQ)
    test_metrics_res18_fe = test(resnet18_frozen, dataloader_test)
    save_model(resnet18_frozen, "resnet18_frozen.pt")

    optimizer = optim.Adam(resnet18_finetune.parameters(), lr=LR)
    train_metrics_res18_ft = train(EPOCHS, resnet18_finetune, dataloader_train, dataloader_val, optimizer, criterion, VAL_FREQ)
    test_metrics_res18_ft = test(resnet18_finetune, dataloader_test)
    save_model(resnet18_finetune, "resnet18_finetune.pt")

    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
