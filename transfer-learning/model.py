# model.py

import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, num_classes=102):
        super(CNN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 8, 3),
            nn.BatchNorm2d(8),  
            nn.ReLU(),
            nn.Conv2d(8, 16, 3),
            nn.BatchNorm2d(16),  
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3),
            nn.BatchNorm2d(64),  
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(238144, 4096),
            nn.BatchNorm1d(4096), 
            nn.ReLU(),
            nn.Linear(4096, 1000),
            nn.BatchNorm1d(1000), 
            nn.ReLU(),
            nn.Linear(1000, num_classes),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x
