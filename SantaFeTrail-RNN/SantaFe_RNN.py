import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque

class SantaFeRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SantaFeRNN, self).__init__()
        channels, height, width = input_size
        # --- CNN feature extractor (copied from SantaFeCNN) ---
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Calculate flattened CNN feature size
        cnn_flattened_size = 128 * (height // 8) * (width // 8)
        # --- RNN ---
        self.rnn = nn.RNN(input_size=cnn_flattened_size, hidden_size=hidden_size, batch_first=True)
        # --- FC layers ---
        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        # x shape: (batch_size, seq_len, channels, height, width)
        batch_size, seq_len, channels, height, width = x.size()
        # Reshape to (batch_size * seq_len, channels, height, width)
        x = x.view(batch_size * seq_len, channels, height, width)
        # Pass each frame through CNN
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)  # (batch_size * seq_len, cnn_flattened_size)
        # Reshape to (batch_size, seq_len, cnn_flattened_size)
        x = x.view(batch_size, seq_len, -1)
        # Pass through RNN
        x, _ = self.rnn(x)
        x = x[:, -1, :]  # Take last time step
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
# Registering custom environment
gym.register(
        id="gymnasium_env/SantaFeTrail-v0",
        entry_point="SantaFeTrailEnv:SantaFeTrailEnv",  
        reward_threshold=89,        
        max_episode_steps=150     
)

# Initializing environment
env = gym.make("gymnasium_env/SantaFeTrail-v0")

# Hyperparameters
num_episodes = 500
batch_size = 64
gamma = 0.9006346496123904
epsilon_start = 1.0
epsilon_end = 0.1
epsilon_decay = 0.9910256339861461
learning_rate = 0.0011779172637528985
replay_buffer = deque(maxlen=50000)
recent_buffer = deque(maxlen=5000)  # smaller buffer for recent transitions
