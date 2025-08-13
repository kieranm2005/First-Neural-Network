import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 16)
        self.fc2 = nn.Linear(16, 1)  # Output: input current (0–255)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # Range: [0, 1]
        return x * 255  # Scale to [0, 255] for FPGA input
