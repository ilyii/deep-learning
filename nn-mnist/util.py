from collections import defaultdict
import random

import numpy as np
import torch


class Logger:

    def __init__(self, save_path:str = None):
        self.path = save_path
        self.data = defaultdict(list)

    def log(self, data: dict):
        for key, value in data.items():
            self.data[key].append(value)


def set_seed(seed):
    random(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)