import snntorch as snn
from snntorch import surrogate, backprop, functional as SF, utils, spikeplot as splt

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import itertools

spike_grad = surrogate.fast_sigmoid(slope=25)
beta = 0.5
num_steps = 50

lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)


dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


# Architecture adapted from SantaFe_CNN.py, using snn.Leaky where appropriate

net = nn.Sequential(
    nn.Conv2d(1, 32, 3, padding=1),  # Conv layer, 1 input channel, 32 output channels, 3x3 kernel
    nn.ReLU(),
    nn.MaxPool2d(2),                 # 14x14
    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),

    nn.Conv2d(32, 64, 3, padding=1), # Conv layer, 32 input, 64 output, 3x3 kernel
    nn.ReLU(),
    nn.MaxPool2d(2),                 # 7x7
    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),

    nn.Flatten(),
    nn.Linear(64 * 7 * 7, 128),
    nn.ReLU(),
    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),

    nn.Linear(128, 10),
    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
).to(device)