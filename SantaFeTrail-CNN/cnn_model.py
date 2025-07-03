import torch
import torch.nn as nn
import torch.nn.functional as F

class SantaFeCNN(nn.Module):
    def __init__(self, observation_shape, num_actions):
        super(SantaFeCNN, self).__init__()
        # Accepts either (C, H, W) or (N,) where N = C*H*W
        if len(observation_shape) == 1:
            self.channels, self.height, self.width = 6, 32, 32  # Set to intended values if flat
        else:
            self.channels, self.height, self.width = observation_shape

        self.conv1 = nn.Conv2d(self.channels, 32, 3, 1, 1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.pool3 = nn.MaxPool2d(2, 2)
        flattened_size = 128 * (self.height // 8) * (self.width // 8)
        self.fc1 = nn.Linear(flattened_size, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
        if x.dim() == 2 and x.shape[1] == self.channels * self.height * self.width:
            x = x.view(x.size(0), self.channels, self.height, self.width)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
