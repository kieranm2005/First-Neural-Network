# imports
import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Temporal dynamics
alpha = 0.9
beta = 0.8
num_steps = 200

# Initialize 2nd-order LIF neuron
lif1 = snn.Synaptic(alpha=alpha, beta=beta)

# Periodic spiking input, spk_in = 0.2 V
w = 0.2
spk_period = torch.cat((torch.ones(1)*w, torch.zeros(9)), 0)
spk_in = spk_period.repeat(20)

# Initialize hidden states and output
syn, mem = lif1.init_synaptic()
spk_out = torch.zeros(1)
syn_rec = []
mem_rec = []
spk_rec = []

# Simulate neurons
for step in range(num_steps):
  spk_out, syn, mem = lif1(spk_in[step], syn, mem)
  spk_rec.append(spk_out)
  syn_rec.append(syn)
  mem_rec.append(mem)

# convert lists to tensors
spk_rec = torch.stack(spk_rec)
syn_rec = torch.stack(syn_rec)
mem_rec = torch.stack(mem_rec)

def plot_spk_cur_mem_spk(spk_in, syn_rec, mem_rec, spk_rec, title):
    fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    axs[0].plot(spk_in.detach().cpu().numpy())
    axs[0].set_ylabel("Input")
    axs[1].plot(syn_rec.detach().cpu().numpy())
    axs[1].set_ylabel("Synaptic\nCurrent")
    axs[2].plot(mem_rec.detach().cpu().numpy())
    axs[2].set_ylabel("Membrane\nPotential")
    axs[3].plot(spk_rec.detach().cpu().numpy())
    axs[3].set_ylabel("Output\nSpikes")
    axs[3].set_xlabel("Time step")
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

plot_spk_cur_mem_spk(spk_in, syn_rec, mem_rec, spk_rec, "Synaptic Conductance-based Neuron Model With Input Spikes")

alpha = 0.8
beta = 0.7

# initialize neuron
lif2 = snn.Alpha(alpha=alpha, beta=beta, threshold=0.5)

# input spike: initial spike, and then period spiking
w = 0.85
spk_in = (torch.cat((torch.zeros(10), torch.ones(1), torch.zeros(89),
                     (torch.cat((torch.ones(1), torch.zeros(9)),0).repeat(10))), 0) * w).unsqueeze(1)

# initialize parameters
syn_exc, syn_inh, mem = lif2.init_alpha()
mem_rec = []
spk_rec = []

# run simulation
for step in range(num_steps):
  spk_out, syn_exc, syn_inh, mem = lif2(spk_in[step], syn_exc, syn_inh, mem)
  mem_rec.append(mem.squeeze(0))
  spk_rec.append(spk_out.squeeze(0))

# convert lists to tensors
mem_rec = torch.stack(mem_rec)
spk_rec = torch.stack(spk_rec)

def plot_spk_mem_spk(spk_in, mem_rec, spk_rec, title):
    fig, axs = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    axs[0].plot(spk_in.detach().cpu().numpy())
    axs[0].set_ylabel("Input")
    axs[1].plot(mem_rec.detach().cpu().numpy())
    axs[1].set_ylabel("Membrane\nPotential")
    axs[2].plot(spk_rec.detach().cpu().numpy())
    axs[2].set_ylabel("Output\nSpikes")
    axs[2].set_xlabel("Time step")
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

plot_spk_mem_spk(spk_in, mem_rec, spk_rec, "Alpha Neuron Model With Input Spikes")


