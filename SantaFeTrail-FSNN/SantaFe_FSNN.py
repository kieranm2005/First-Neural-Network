import snntorch as snn
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
from collections import deque
import random
from SantaFeTrailEnv import SantaFeTrailEnv
import torch.nn.functional as F
import math

# Fractional LIF neuron model for SNNs
class FractionalLIF(nn.Module):
    def __init__(self, size, alpha=0.7, threshold=1.0, reset=0.0): 
        super().__init__()
        self.size = size
        self.alpha = alpha  # Fractional order (0 < alpha <= 1)
        self.threshold = threshold
        self.reset = reset

    def forward(self, input, mem_hist=None):
        # mem_hist: [T, batch, size] or None
        if mem_hist is None:
            mem_hist = torch.zeros((1, input.shape[0], self.size), device=input.device)
        T = mem_hist.shape[0]
        # Grünwald–Letnikov coefficients
        coeffs = [(-1)**k * math.comb(self.alpha, k) for k in range(T)] # Combinatorial coefficients for fractional order
        coeffs = torch.tensor(coeffs, device=input.device).view(-1, 1, 1) # Shape: [T, 1, 1] for broadcasting
        # Fractional sum
        mem = (coeffs * mem_hist).sum(dim=0) + input 
        spk = (mem >= self.threshold).float() # Spike generation based on threshold
        mem = mem * (1 - spk) + self.reset * spk # Reset membrane potential after spiking
        # Update mem_hist for next step
        mem_hist = torch.cat([mem.unsqueeze(0), mem_hist[:-1]], dim=0)
        return spk, mem, mem_hist

# Define the SNN model
class SNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SNNModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.lif1 = FractionalLIF(hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.lif2 = FractionalLIF(output_size)

    def forward(self, x, mem_hist=None):
        x = F.relu(self.fc1(x))
        spk1, mem1, mem_hist = self.lif1(x, mem_hist)
        x = F.relu(self.fc2(spk1))
        spk2, mem2, _ = self.lif2(x)
        return spk2, mem_hist
    

class SantaFeTrailSNN(SNNModel):
    def __init__(self, observation_shape, num_actions):
        super(SantaFeTrailSNN, self).__init__(input_size=np.prod(observation_shape), hidden_size=16, output_size=num_actions)
        self.observation_shape = observation_shape
        self.num_actions = num_actions

    def forward(self, x, mem_hist=None):
        # Flatten the input
        x = x.view(x.size(0), -1)  # Flatten the input tensor
        spk2, mem_hist = super().forward(x, mem_hist)
        return spk2
    
    # Registering custom environment
    def register_initialize_env(self):
        gym.register(
        id="gymnasium_env/SantaFeTrail-v0",
        entry_point="SantaFeTrailEnv:SantaFeTrailEnv",  # module:class
        reward_threshold=89,        
        max_episode_steps=600      # Increased from 100 to allow more moves
    )
        self.env = gym.make("gymnasium_env/SantaFeTrail-v0")
        return self.env
    
class SantaFeTrailSNNTrainer:
    def __init__(self, model, env, num_episodes=500, batch_size=64, gamma=0.9, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.99, learning_rate=0.001):
        self.model = model
        self.env = env
        self.num_episodes = num_episodes
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.replay_buffer = deque(maxlen=50000)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def train(self):
        