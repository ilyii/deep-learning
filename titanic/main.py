from collections import defaultdict
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import logging
import yaml
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

import util
from model import NN, train, test
from dataset import get_data

logger = logging.getLogger(__name__)

def load(path:str, **kwargs):
    func = f'read_{path.split(".")[-1]}'
    data = getattr(pd, func)(path)

def impute(data:pd.DataFrame, split:str):
    if split == "train":
        cat_features = data.select_dtypes(include=["object"]).columns
        imp_= SimpleImputer(strategy="most_frequent")
        data[cat_features] = imp_.fit_transform(data[cat_features])

        num_features = data.select_dtypes(include=["number"]).columns
        imp_ = SimpleImputer(strategy="mean")
        data[num_features] = imp_.fit_transform(data[num_features])
    else:
        data.dropna(inplace=True)
    return data   



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
    dataset_train, dataloader_train = get_data(opt, split="train")
    dataset_test, dataloader_test = get_data(opt, split="test")

    print(dataset_train.X.shape, dataset_train.y.shape, dataset_test.X.shape, dataset_test.y.shape)


    # Model
    archs = opt.archs
    print(archs)
    models = [NN(in_dim=dataset_train.X.shape[1], **arch) for k, arch in archs.items()]

    # Training
    stats = defaultdict(dict)
    for idx, model in enumerate(models):
        model = model.to(opt.device)
        optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum)
        criterion = nn.BCELoss()
        stats[archs.keys()[idx]] = train(model=model,
                                         dataloader_train=dataloader_train,
                                         dataloader_test=dataloader_test,
                                         optimizer=optimizer,
                                         criterion=criterion,
                                         **opt)


    # Plot all losses and accuracies in 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for key, value in stats.items():
        sns.lineplot(x=value["epoch"], y=value["loss"], ax=axes[0], label=key)
        sns.lineplot(x=value["epoch"], y=value["acc"], ax=axes[1], label=key)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(opt.savepath, "stats.png"))

    logger.info(f"Done. (Time: {time.time() - init_time:.4f}s)")

    
if __name__ == "__main__":
    with open("options.yaml", "r") as file:
        opt = util.DotDict(yaml.safe_load(file))

    main(opt)