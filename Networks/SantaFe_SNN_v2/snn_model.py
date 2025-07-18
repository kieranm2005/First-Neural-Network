import torch
import torch.nn as nn
import snntorch as snn
import snntorch.functional as SF
import snntorch.spikegen as spikegen

class SantaFeSNN(nn.Module):
    def __init__(self, input_shape, hidden_size, num_actions, beta=0.9, threshold=1.0):
        super().__init__()
        input_dim = input_shape[0]
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold)
        self.fc2 = nn.Linear(hidden_size, num_actions)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.fc1.weight, a=0.01)
        nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_uniform_(self.fc2.weight, a=0.01)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        # x: (batch, seq, input_dim) or (batch, input_dim)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, seq=1, input_dim)
        batch, seq, input_dim = x.shape
        mem1 = self.fc1.weight.new_zeros((batch, self.fc1.out_features))
        spk1 = self.fc1.weight.new_zeros((batch, self.fc1.out_features))
        out_spikes = []
        for t in range(seq):
            cur1 = self.fc1(x[:, t, :])
            spk1, mem1 = self.lif1(cur1, mem1)
            out = self.fc2(spk1)
            out_spikes.append(out.unsqueeze(1))
        return torch.cat(out_spikes, dim=1)  # (batch, seq, num_actions)
