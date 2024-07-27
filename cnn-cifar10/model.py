import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 8, 3),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 4096),
            nn.Linear(4096, 1000),
            nn.ReLU(),
            nn.Linear(1000, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x