import numpy as np
import torch

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0

    def add(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float32) if isinstance(state, np.ndarray) else state
        next_state = torch.tensor(next_state, dtype=torch.float32) if isinstance(next_state, np.ndarray) else next_state
        max_priority = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)

        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) < self.capacity:
            probs = self.priorities[:len(self.buffer)]
        else:
            probs = self.priorities

        probs = probs ** self.alpha
        probs = probs / probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        weights = (len(self.buffer) * probs[indices]) ** -self.beta
        weights = weights / weights.max()

        return samples, indices, weights

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
