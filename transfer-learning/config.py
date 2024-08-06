# config.py

import torch

IMG_SIZE = 256
BATCH_SIZE = 4
LR = 0.001
EPOCHS = 10
VAL_FREQ = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_PATH = "output/"
