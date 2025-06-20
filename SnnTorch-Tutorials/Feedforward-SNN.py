import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def leaky_integrate_and_fire(mem, x, w, beta, threshold=1):
    spk = (mem > threshold) # If membrane potential exceeds threshold, spike
    mem = beta * mem + w*x - spk*threshold
    return spk, mem

# set neuronal parameters
delta_t = torch.tensor(1e-3)
tau = torch.tensor(5e-3)
beta = torch.exp(-delta_t/tau)

print(f"The decay rate is: {beta:.3f}")

num_steps = 200

# initialize inputs/outputs + small step current input
x = torch.cat((torch.zeros(10), torch.ones(190)*0.5), 0)
mem = torch.zeros(1)
spk_out = torch.zeros(1)
mem_rec = []
spk_rec = []

# neuron parameters
w = 0.4
beta = 0.819

# neuron simulation
for step in range(num_steps):
  spk, mem = leaky_integrate_and_fire(mem, x[step], w=w, beta=beta)
  mem_rec.append(mem)
  spk_rec.append(spk)

# convert lists to tensors
mem_rec = torch.stack(mem_rec)
spk_rec = torch.stack(spk_rec)

def plot_cur_mem_spk(cur, mem, spk, thr_line=1, ylim_max1=1.0, title=""):
    fig, axs = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    axs[0].plot(cur)
    axs[0].set_title("Input Current")
    axs[0].set_ylim([0, ylim_max1])
    axs[1].plot(mem)
    axs[1].axhline(thr_line, color="r", linestyle="--", label="Threshold")
    axs[1].set_title("Membrane Potential")
    axs[2].plot(spk)
    axs[2].set_title("Output Spikes")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

plot_cur_mem_spk(x*w, mem_rec, spk_rec, thr_line=1,ylim_max1=0.5, title="LIF Neuron Model With Weighted Step Voltage")

lif1 = snn.Leaky(beta=0.8)

# Small step current input
w=0.21
cur_in = torch.cat((torch.zeros(10), torch.ones(190)*w), 0)
mem = torch.zeros(1)
spk = torch.zeros(1)
mem_rec = []
spk_rec = []

# neuron simulation
for step in range(num_steps):
  spk, mem = lif1(cur_in[step], mem)
  mem_rec.append(mem)
  spk_rec.append(spk)

# convert lists to tensors
mem_rec = torch.stack(mem_rec)
spk_rec = torch.stack(spk_rec)

plot_cur_mem_spk(cur_in, mem_rec, spk_rec, thr_line=1, ylim_max1=0.5, title="snn.Leaky Neuron Model")

# layer parameters
num_inputs = 784
num_hidden = 1000
num_outputs = 10
beta = 0.99

# initialize layers
fc1 = nn.Linear(num_inputs, num_hidden)
lif1 = snn.Leaky(beta=beta)
fc2 = nn.Linear(num_hidden, num_outputs)
lif2 = snn.Leaky(beta=beta)

# Initialize hidden states
mem1 = lif1.init_leaky()
mem2 = lif2.init_leaky()

# record outputs
mem2_rec = []
spk1_rec = []
spk2_rec = []

spk_in = spikegen.rate_conv(torch.rand((200, 784))).unsqueeze(1)
print(f"Dimensions of spk_in: {spk_in.size()}")

# network simulation
for step in range(num_steps):
    cur1 = fc1(spk_in[step]) # post-synaptic current <-- spk_in x weight
    spk1, mem1 = lif1(cur1, mem1) # mem[t+1] <--post-syn current + decayed membrane
    cur2 = fc2(spk1)
    spk2, mem2 = lif2(cur2, mem2)

    mem2_rec.append(mem2)
    spk1_rec.append(spk1)
    spk2_rec.append(spk2)

# convert lists to tensors
mem2_rec = torch.stack(mem2_rec)
spk1_rec = torch.stack(spk1_rec)
spk2_rec = torch.stack(spk2_rec)

def plot_snn_spikes(spk_in, spk1_rec, spk2_rec, title=""):
    fig, axs = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    axs[0].imshow(spk_in.cpu().squeeze().t(), aspect="auto", cmap="Greys")
    axs[0].set_title("Input Spikes")
    axs[1].imshow(spk1_rec.cpu().detach().squeeze().t(), aspect="auto", cmap="Greys")
    axs[1].set_title("Hidden Layer Spikes")
    axs[2].imshow(spk2_rec.cpu().detach().squeeze().t(), aspect="auto", cmap="Greys")
    axs[2].set_title("Output Layer Spikes")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

plot_snn_spikes(spk_in, spk1_rec, spk2_rec, "Fully Connected Spiking Neural Network")

from IPython.display import HTML

fig, ax = plt.subplots(facecolor='w', figsize=(12, 7))
labels=['0', '1', '2', '3', '4', '5', '6', '7', '8','9']
spk2_rec = spk2_rec.squeeze(1).detach().cpu()

plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

#  Plot spike count histogram
# anim = splt.spike_count(spk2_rec, fig, ax, labels=labels, animate=True)
# HTML(anim.to_html5_video())
# anim.save("spike_bar.mp4")

# plot membrane potential traces
splt.traces(mem2_rec.squeeze(1), spk=spk2_rec.squeeze(1))
fig = plt.gcf()
fig.set_size_inches(8, 6)
plt.show()
