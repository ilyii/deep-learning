import os
import random
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision 
from torch.utils.data import DataLoader, Dataset

from model import NN, train, test
import util


logger = logging.getLogger(__name__)


def main(opt):
    init_time = time.time()
    logger.info("Initializing...")
    logger.info(opt)

    util.set_seed(opt.seed)
    os.makedirs(opt.savepath, exist_ok=True)


    logger.info("PyTorch Version: ",torch.__version__)
    logger.info("CUDA is available" if torch.cuda.is_available() else "CUDA is not available")
    for i in range(torch.cuda.device_count()):
        logger.info(torch.cuda.get_device_properties(i).name)

    # Data
    transform = util.get_transform(opt.mean, opt.std)
    dataset_train, dataloader_train = util.get_data(opt, "train", transform)
    dataset_test, dataloader_test = util.get_data(opt, "test", transform)

    # Display
    plt.figure(figsize=(15, 5))
    for i in range(n:=9):
        idx = random.randint(0, len(dataset_train))
        plt.subplot(1,n,i+1)
        img = util.unnorm(dataset_train[idx][0], opt.mean, opt.std)
        plt.imshow(img, cmap='gray')
        plt.title(f"{dataset_train[idx][1]}")
        plt.tight_layout()
        plt.axis('off')
    plt.savefig(os.path.join(opt.savepath, "examples.png"))

    # Model
    model = NN(opt.in_dim, opt.hidden_dims, opt.out_dim).to(opt.device)

    # Training
    optimizer = optim.SGD(model.parameters(), lr=opt.learning_rate, momentum=opt.momentum)
    criterion = nn.NLLLoss()

    s_time = time.time()
    # Train
    
    stats = train(model=model,
                  dataloader_train=dataloader_train,
                  dataloader_test=dataloader_test,
                  optimizer=optimizer,
                  criterion=criterion,
                  **opt)

    logger.info(f"Training Time: {time.time()-s_time:.2f} s")

    util.plot_stats(stats, savepath=opt.savepath)
    
    logger.info("Testing...")

    # Test
    loss, acc = test(model, dataloader_test, criterion, opt.device)
    logger.info(f"----------- Results -----------")
    logger.info(f"Trained epochs: {opt.epochs}")
    logger.info(f"Test Loss: {loss:.4f}")
    logger.info(f"Test Accuracy: {acc:.4f}")
    logger.info(f"-------------------------------")

    logger.info(f"Done. ({time.time()-init_time:.2f} s)")


if __name__ == "__main__":
    with open("options.yaml", "r") as file:
        opt = util.DotDict(yaml.safe_load(file))

    main(opt)

