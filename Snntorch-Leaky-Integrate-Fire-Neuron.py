import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen

import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt

def leaky_integrate_neuron(U, time_step=1e-3, I=0, R=5e7, C=1e-10):
  tau = R*C
  U = U + (time_step/tau)*(-U + I*R)
  return U

num_steps = 100
U = 0.9
U_trace = []  # keeps a record of U for plotting

for step in range(num_steps):
  U_trace.append(U)
  U = leaky_integrate_neuron(U)  # solve next step of U

def plot_mem(U_trace, title="Membrane Potential Trace"):
    plt.figure(figsize=(8, 4))
    plt.plot(U_trace)
    plt.title(title)
    plt.xlabel("Time step")
    plt.ylabel("Membrane Potential (U)")
    plt.grid(True)
    plt.show()

plot_mem(U_trace, "Leaky Neuron Model")


time_step = 1e-3  # 1 ms time step
R = 5
C = 1e-3  # 1 mF capacitance

# Leaky Integrate and Fire neuron, tau= R * C, 5e-3
lif1 = snn.Lapicque(R=R, C=C, time_step=time_step)

#Initialize membrane, input, and output
mem = torch.ones(1) * 0.9 # U=0.9 at t=0
cur_in = torch.zeros(num_steps, 1) # I=0 for all t
spk_out = torch.zeros(1)  # initialize output spikes

# A list to store a recording of membrane potential
mem_rec = [mem]

# pass updated value of mem and cur_in[step]=0 at every time step
for step in range(num_steps):
  spk_out, mem = lif1(cur_in[step], mem)

  # Store recordings of membrane potential
  mem_rec.append(mem)

# convert the list of tensors into one tensor
mem_rec = torch.stack(mem_rec)

# pre-defined plotting function
plot_mem(mem_rec, "Lapicque's Neuron Model Without Stimulus")

# Initialize input current pulse
cur_in = torch.cat((torch.zeros(10, 1), torch.ones(190, 1)*0.1), 0)  # input current turns on at t=10

# Initialize membrane, output and recordings
mem = torch.zeros(1)  # membrane potential of 0 at t=0
spk_out = torch.zeros(1)  # neuron needs somewhere to sequentially dump its output spikes
mem_rec = [mem]

num_steps = 200

# pass updated value of mem and cur_in[step] at every time step
for step in range(num_steps):
  spk_out, mem = lif1(cur_in[step], mem)
  mem_rec.append(mem)

# crunch -list- of tensors into one tensor
mem_rec = torch.stack(mem_rec)

def plot_step_current_response(cur_in, mem_rec, step_on, title="Step Current Response"):
    plt.figure(figsize=(10, 5))
    plt.subplot(2, 1, 1)
    plt.plot(cur_in.numpy(), label="Input Current")
    plt.axvline(x=step_on, color='r', linestyle='--', label="Step Onset")
    plt.ylabel("Current (I)")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(mem_rec.numpy(), label="Membrane Potential")
    plt.axvline(x=step_on, color='r', linestyle='--', label="Step Onset")
    plt.xlabel("Time step")
    plt.ylabel("Membrane Potential (U)")
    plt.legend()
    plt.grid(True)
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

plot_step_current_response(cur_in, mem_rec, 10)

print(f"The calculated value of input pulse [A] x resistance [Ω] is: {cur_in[11]*lif1.R} V")
print(f"The simulated value of steady-state membrane potential is: {mem_rec[200][0]} V")

# Pulse input

# Initialize current pulse, membrane and outputs
cur_in1 = torch.cat((torch.zeros(10, 1), torch.ones(20, 1)*(0.1), torch.zeros(170, 1)), 0)  # input turns on at t=10, off at t=30
mem = torch.zeros(1)
spk_out = torch.zeros(1)
mem_rec1 = [mem]

# neuron simulation
for step in range(num_steps):
  spk_out, mem = lif1(cur_in1[step], mem)
  mem_rec1.append(mem)
mem_rec1 = torch.stack(mem_rec1)

def plot_current_pulse_response(cur_in, mem_rec, title="Current Pulse Response", line1=None, vline2=None):
    plt.figure(figsize=(10, 5))
    plt.subplot(2, 1, 1)
    plt.plot(cur_in.numpy(), label="Input Current")
    if line1 is not None:
        plt.axvline(x=line1, color='r', linestyle='--', label="Pulse Onset")
    if vline2 is not None:
        plt.axvline(x=vline2, color='g', linestyle='--', label="Pulse Offset")
    plt.ylabel("Current (I)")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(mem_rec.numpy(), label="Membrane Potential")
    if line1 is not None:
        plt.axvline(x=line1, color='r', linestyle='--', label="Pulse Onset")
    if vline2 is not None:
        plt.axvline(x=vline2, color='g', linestyle='--', label="Pulse Offset")
    plt.xlabel("Time step")
    plt.ylabel("Membrane Potential (U)")
    plt.legend()
    plt.grid(True)
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

plot_current_pulse_response(cur_in1, mem_rec1, "Lapicque's Neuron Model With Input Pulse", line1=10, vline2=30)

# Deliver the same amount of charge in half the time

# Increase amplitude of current pulse; half the time
cur_in2 = torch.cat((torch.zeros(10, 1), torch.ones(10, 1)*0.111, torch.zeros(180, 1)), 0)  # input turns on at t=10, off at t=20
mem = torch.zeros(1)
spk_out = torch.zeros(1)
mem_rec2 = [mem]

# neuron simulation
for step in range(num_steps):
  spk_out, mem = lif1(cur_in2[step], mem)
  mem_rec2.append(mem)
mem_rec2 = torch.stack(mem_rec2)

plot_current_pulse_response(cur_in2, mem_rec2, "Lapicque's Neuron Model With Input Pulse: x1/2 pulse width", line1=10, vline2=20)

# Again, with a faster pulse

# Increase amplitude of current pulse; quarter the time.
cur_in3 = torch.cat((torch.zeros(10, 1), torch.ones(5, 1)*0.147, torch.zeros(185, 1)), 0)  # input turns on at t=10, off at t=15
mem = torch.zeros(1)
spk_out = torch.zeros(1)
mem_rec3 = [mem]

# neuron simulation
for step in range(num_steps):
  spk_out, mem = lif1(cur_in3[step], mem)
  mem_rec3.append(mem)
mem_rec3 = torch.stack(mem_rec3)

plot_current_pulse_response(cur_in3, mem_rec3, "Lapicque's Neuron Model With Input Pulse: x1/4 pulse width", line1=10, vline2=15)

# Compare all three on the same plot

def compare_plots(cur_in1, cur_in2, cur_in3, mem_rec1, mem_rec2, mem_rec3, pulse1_on, pulse1_off, pulse2_off, pulse3_off, title):
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 2, 1)
    plt.plot(cur_in1.numpy(), label="Input 1")
    plt.axvline(x=pulse1_on, color='r', linestyle='--')
    plt.axvline(x=pulse1_off, color='g', linestyle='--')
    plt.ylabel("Current (I)")
    plt.title("Current Pulse 1")
    plt.grid(True)

    plt.subplot(3, 2, 2)
    plt.plot(mem_rec1.numpy(), label="Membrane 1")
    plt.axvline(x=pulse1_on, color='r', linestyle='--')
    plt.axvline(x=pulse1_off, color='g', linestyle='--')
    plt.ylabel("Membrane Potential (U)")
    plt.title("Membrane Response 1")
    plt.grid(True)

    plt.subplot(3, 2, 3)
    plt.plot(cur_in2.numpy(), label="Input 2")
    plt.axvline(x=pulse1_on, color='r', linestyle='--')
    plt.axvline(x=pulse2_off, color='g', linestyle='--')
    plt.ylabel("Current (I)")
    plt.title("Current Pulse 2")
    plt.grid(True)

    plt.subplot(3, 2, 4)
    plt.plot(mem_rec2.numpy(), label="Membrane 2")
    plt.axvline(x=pulse1_on, color='r', linestyle='--')
    plt.axvline(x=pulse2_off, color='g', linestyle='--')
    plt.ylabel("Membrane Potential (U)")
    plt.title("Membrane Response 2")
    plt.grid(True)

    plt.subplot(3, 2, 5)
    plt.plot(cur_in3.numpy(), label="Input 3")
    plt.axvline(x=pulse1_on, color='r', linestyle='--')
    plt.axvline(x=pulse3_off, color='g', linestyle='--')
    plt.xlabel("Time step")
    plt.ylabel("Current (I)")
    plt.title("Current Pulse 3")
    plt.grid(True)

    plt.subplot(3, 2, 6)
    plt.plot(mem_rec3.numpy(), label="Membrane 3")
    plt.axvline(x=pulse1_on, color='r', linestyle='--')
    plt.axvline(x=pulse3_off, color='g', linestyle='--')
    plt.xlabel("Time step")
    plt.ylabel("Membrane Potential (U)")
    plt.title("Membrane Response 3")
    plt.grid(True)

    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

compare_plots(cur_in1, cur_in2, cur_in3, mem_rec1, mem_rec2, mem_rec3, 10, 15, 20, 30, "Lapicque's Neuron Model With Input Pulse: Varying inputs")

# Current spike input
cur_in4 = torch.cat((torch.zeros(10, 1), torch.ones(1, 1)*0.5, torch.zeros(189, 1)), 0)  # input only on for 1 time step
mem = torch.zeros(1)
spk_out = torch.zeros(1)
mem_rec4 = [mem]

# neuron simulation
for step in range(num_steps):
  spk_out, mem = lif1(cur_in4[step], mem)
  mem_rec4.append(mem)
mem_rec4 = torch.stack(mem_rec4)

plot_current_pulse_response(cur_in4, mem_rec4, "Lapicque's Neuron Model With Input Spike", line1=10, ylim_max1=0.6)

# R=5.1, C=5e-3 for illustrative purposes
# LIF w/Reset mechanism
def leaky_integrate_and_fire(mem, cur=0, threshold=1, time_step=1e-3, R=5.1, C=5e-3):
  tau_mem = R*C
  spk = (mem > threshold)
  mem = mem + (time_step/tau_mem)*(-mem + cur*R) - spk*threshold  # every time spk=1, subtract the threhsold
  return mem, spk

# Small step current input
cur_in = torch.cat((torch.zeros(10), torch.ones(190)*0.2), 0)
mem = torch.zeros(1)
mem_rec = []
spk_rec = []

# neuron simulation
for step in range(num_steps):
  mem, spk = leaky_integrate_and_fire(mem, cur_in[step])
  mem_rec.append(mem)
  spk_rec.append(spk)

# convert lists to tensors
mem_rec = torch.stack(mem_rec)
spk_rec = torch.stack(spk_rec)

def plot_cur_mem_spk(cur_in, mem_rec, spk_rec, thr_line=None, vline=None, ylim_max2=None, title=""):
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(cur_in.numpy(), label="Input Current")
    if vline is not None:
        plt.axvline(x=vline, color='r', linestyle='--', label="Spike Onset")
    plt.ylabel("Current (I)")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(mem_rec.numpy(), label="Membrane Potential")
    if thr_line is not None:
        plt.axhline(y=thr_line, color='g', linestyle='--', label="Threshold")
    if vline is not None:
        plt.axvline(x=vline, color='r', linestyle='--')
    if ylim_max2 is not None:
        plt.ylim([0, ylim_max2])
    plt.ylabel("Membrane Potential (U)")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(spk_rec.numpy(), label="Spikes")
    if vline is not None:
        plt.axvline(x=vline, color='r', linestyle='--')
    plt.xlabel("Time step")
    plt.ylabel("Spike")
    plt.legend()
    plt.grid(True)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

plot_cur_mem_spk(cur_in, mem_rec, spk_rec, thr_line=1, vline=109, ylim_max2=1.3, title="LIF Neuron Model With Reset")

# Create the same neuron as before using snnTorch
lif2 = snn.Lapicque(R=5.1, C=5e-3, time_step=1e-3)

print(f"Membrane potential time constant: {lif2.R * lif2.C:.3f}s")

# Initialize inputs and outputs
cur_in = torch.cat((torch.zeros(10, 1), torch.ones(190, 1)*0.2), 0)
mem = torch.zeros(1)
spk_out = torch.zeros(1)
mem_rec = [mem]
spk_rec = [spk_out]

# Simulation run across 100 time steps.
for step in range(num_steps):
  spk_out, mem = lif2(cur_in[step], mem)
  mem_rec.append(mem)
  spk_rec.append(spk_out)

# convert lists to tensors
mem_rec = torch.stack(mem_rec)
spk_rec = torch.stack(spk_rec)

plot_cur_mem_spk(cur_in, mem_rec, spk_rec, thr_line=1, vline=109, ylim_max2=1.3, title="Lapicque Neuron Model With Step Input")

print(spk_rec[105:115].view(-1))

# Initialize inputs and outputs
cur_in = torch.cat((torch.zeros(10, 1), torch.ones(190, 1)*0.3), 0)  # increased current
mem = torch.zeros(1)
spk_out = torch.zeros(1)
mem_rec = [mem]
spk_rec = [spk_out]

# neuron simulation
for step in range(num_steps):
  spk_out, mem = lif2(cur_in[step], mem)
  mem_rec.append(mem)
  spk_rec.append(spk_out)

# convert lists to tensors
mem_rec = torch.stack(mem_rec)
spk_rec = torch.stack(spk_rec)


plot_cur_mem_spk(cur_in, mem_rec, spk_rec, thr_line=1, ylim_max2=1.3, title="Lapicque Neuron Model With Periodic Firing")

# neuron with halved threshold
lif3 = snn.Lapicque(R=5.1, C=5e-3, time_step=1e-3, threshold=0.5)

# Initialize inputs and outputs
cur_in = torch.cat((torch.zeros(10, 1), torch.ones(190, 1)*0.3), 0)
mem = torch.zeros(1)
spk_out = torch.zeros(1)
mem_rec = [mem]
spk_rec = [spk_out]

# Neuron simulation
for step in range(num_steps):
  spk_out, mem = lif3(cur_in[step], mem)
  mem_rec.append(mem)
  spk_rec.append(spk_out)

# convert lists to tensors
mem_rec = torch.stack(mem_rec)
spk_rec = torch.stack(spk_rec)

plot_cur_mem_spk(cur_in, mem_rec, spk_rec, thr_line=0.5, ylim_max2=1.3, title="Lapicque Neuron Model With Lower Threshold")

# Create a 1-D random spike train. Each element has a probability of 40% of firing.
spk_in = spikegen.rate_conv(torch.ones((num_steps,1)) * 0.40)

print(f"There are {int(sum(spk_in))} total spikes out of {len(spk_in)} time steps.")

fig = plt.figure(facecolor="w", figsize=(8, 1))
ax = fig.add_subplot(111)

splt.raster(spk_in.reshape(num_steps, -1), ax, s=100, c="black", marker="|")
plt.title("Input Spikes")
plt.xlabel("Time step")
plt.yticks([])
plt.show()

# Initialize inputs and outputs
mem = torch.ones(1)*0.5
spk_out = torch.zeros(1)
mem_rec = [mem]
spk_rec = [spk_out]

# Neuron simulation
for step in range(num_steps):
  spk_out, mem = lif3(spk_in[step], mem)
  spk_rec.append(spk_out)
  mem_rec.append(mem)

# convert lists to tensors
mem_rec = torch.stack(mem_rec)
spk_rec = torch.stack(spk_rec)

def plot_spk_mem_spk(spk_in, mem_rec, spk_rec, title=""):
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(spk_in.numpy(), label="Input Spikes")
    plt.ylabel("Input Spikes")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(mem_rec.numpy(), label="Membrane Potential")
    plt.ylabel("Membrane Potential (U)")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(spk_rec.numpy(), label="Output Spikes")
    plt.xlabel("Time step")
    plt.ylabel("Output Spikes")
    plt.legend()
    plt.grid(True)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

plot_spk_mem_spk(spk_in, mem_rec, spk_rec, "Lapicque's Neuron Model With Input Spikes")

# Neuron with reset_mechanism set to "zero"
lif4 = snn.Lapicque(R=5.1, C=5e-3, time_step=1e-3, threshold=0.5, reset_mechanism="zero")

# Initialize inputs and outputs
spk_in = spikegen.rate_conv(torch.ones((num_steps, 1)) * 0.40)
mem = torch.ones(1)*0.5
spk_out = torch.zeros(1)
mem_rec0 = [mem]
spk_rec0 = [spk_out]

# Neuron simulation
for step in range(num_steps):
  spk_out, mem = lif4(spk_in[step], mem)
  spk_rec0.append(spk_out)
  mem_rec0.append(mem)

# convert lists to tensors
mem_rec0 = torch.stack(mem_rec0)
spk_rec0 = torch.stack(spk_rec0)

def plot_reset_comparison(spk_in, mem_rec, spk_rec, mem_rec0, spk_rec0, title="Reset Mechanism Comparison"):
    plt.figure(figsize=(12, 10))

    plt.subplot(4, 1, 1)
    plt.plot(spk_in.numpy(), label="Input Spikes")
    plt.ylabel("Input Spikes")
    plt.legend()
    plt.grid(True)

    plt.subplot(4, 1, 2)
    plt.plot(mem_rec.numpy(), label="Membrane (Subtract Threshold)")
    plt.plot(mem_rec0.numpy(), label="Membrane (Reset Zero)", linestyle='--')
    plt.ylabel("Membrane Potential (U)")
    plt.legend()
    plt.grid(True)

    plt.subplot(4, 1, 3)
    plt.plot(spk_rec.numpy(), label="Output Spikes (Subtract Threshold)")
    plt.ylabel("Output Spikes")
    plt.legend()
    plt.grid(True)

    plt.subplot(4, 1, 4)
    plt.plot(spk_rec0.numpy(), label="Output Spikes (Reset Zero)", linestyle='--')
    plt.xlabel("Time step")
    plt.ylabel("Output Spikes")
    plt.legend()
    plt.grid(True)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

plot_reset_comparison(spk_in, mem_rec, spk_rec, mem_rec0, spk_rec0)




