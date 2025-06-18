import snntorch as snn
import torch

# Training parameters
batch_size = 128
data_path = '/tmp/data/mnist'
num_epochs = 10 # Because MNIST has 10 output classes

# Torch variables
dtype = torch.float

from torchvision import datasets, transforms # Importing datasets and transforms from torchvision

# Define a transform, which converts images to tensors and normalizes them
# AKA the preprocessing step
transform = transforms.Compose([
    transforms.Resize((28, 28)),  # Resize images to 28x28 pixels
    transforms.Grayscale(),  # Convert images to grayscale
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize((0,), (1,))  # Normalize the images
])

# Load the MNIST dataset
mnist_train = datasets.MNIST(root=data_path, train=True, download=True, transform=transform)

from snntorch import utils # Utils has some useful functions for modifying datasets

# Reduce the dataset by a factor of 10
subset = 10
mnist_train = utils.data_subset(mnist_train, subset)
# Print the size of the dataset
print(f"Size of the dataset: {len(mnist_train)}")

from torch.utils.data import DataLoader # Dataloader is used to load the dataset objects in batches

train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)

# Temporal dynamics
num_steps = 10

# Create vector filled with 0.5
raw_vector = torch.ones(num_steps)*0.5

# Pass each sample through a Bernoulli trial
rate_coded_vector = torch.bernoulli(raw_vector)

# Print the rate coded vector
print("Converted Vector:", rate_coded_vector)

# Print the % of time the output is spiking
print(f"The output is spiking {rate_coded_vector.sum()*100/len(rate_coded_vector):.2f}% of the time")

# Again, but increasing the length of raw_vector

num_steps = 100

# Create vector filled with 0.5
raw_vector = torch.ones(num_steps)*0.5

# Pass each sample through a Bernoulli trial
rate_coded_vector = torch.bernoulli(raw_vector)

# Print the rate coded vector
print("Converted Vector:", rate_coded_vector)

# Print the % of time the output is spiking
print(f"The output is spiking {rate_coded_vector.sum()*100/len(rate_coded_vector):.2f}% of the time")

'''As num_steps approaches infinity, the rate coded vector approaches a constant value of 0.5.
This is because the Bernoulli distribution converges to a normal distribution as the number of trials increases.
In this case, the mean of the normal distribution is 0.5, which is the value we used to fill the raw_vector.
Therefore, the rate coded vector will always be close to 0.5, regardless of the number of steps.
This is a key property of rate coding, as it allows us to represent continuous values using discrete spikes.
In practice, we can use a smaller number of steps to approximate the continuous value, while still maintaining a good representation of the input.
This is why we often use a small number of steps in practice, such as 10 or 20, to represent continuous values.
'''

from snntorch import spikegen  # Importing the spike generator from snntorch

# Iterate through minibatches
data = iter(train_loader) # Get an iterator for the training data
data_it, targets_it = next(data) # Get the next batch of data and targets
# The above line will return a batch of data and targets, which are PyTorch tensors

# Spiking data
spike_data = spikegen.rate(data_it, num_steps=num_steps) #This will convert the data to spikes using the rate coding method

# Print the size of the spike data
print(f"Size of the spike data: {spike_data.size()}")  # Should be [batch_size, num_steps, 1, 28, 28] for MNIST

# Visualizing the data
import matplotlib.pyplot as plt  # Importing matplotlib for plotting
import snntorch.spikeplot as splt # Importing the spike plot module from snntorch
from IPython.display import HTML # Importing HTML for displaying plots

# Plot one sample of data by indexing into a single sample from the batch (B) dimension of spike_data
# [T x B x 1 x 28 x 28] -> [1 x 10 x 1 x 28 x 28]
spike_data_sample = spike_data[:, 0, 0]
# Print the sample
print(f"Sample spike data shape: {spike_data_sample.size()}")  # Should be [num_steps, 28, 28]

torch.Size([num_steps, 28, 28]) # Plot the sample using the spike plot module

fig, ax = plt.subplots()  # Create a figure and axis for plotting
anim = splt.animator(spike_data_sample, fig, ax)
plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg' # Set the path to ffmpeg for saving the animation
HTML(anim.to_html5_video())  # Display the animation as HTML5 video
anim.save("spike_mnist_test.mp4") # Save the animation as an mp4 file
print(f"The corresponding target is: {targets_it[0]}") # Print the target for the sample

# Again, but with spiking frequency of 25%
spike_data = spikegen.rate(data_it, num_steps=num_steps, gain=0.25)  # Set the rate to 0.25
spike_data_sample2 = spike_data[:, 0, 0]  # Get the sample again
fig, ax = plt.subplots()  # Create a figure and axis for plotting
anim = splt.animator(spike_data_sample2, fig, ax)  # Create the animator for the new sample
plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'  # Set the path to ffmpeg for saving the animation
HTML(anim.to_html5_video())  # Display the animation as HTML5 video
anim.save("spike_mnist_test_25.mp4")  # Save the animation as an mp4 file
print(f"The corresponding target is: {targets_it[0]}")  # Print the target for the sample

plt.figure(facecolor="w")  # Create a new figure with a white background
plt.subplot(1,2,1)  # Add a subplot in position 1 of a 1x2 grid
plt.imshow(spike_data_sample.mean(axis=0).reshape((28,-1)).cpu(), cmap='binary')  # Show the mean spike activity for sample 1 as a 28x28 image
plt.axis('off')  # Hide axis ticks and labels
plt.title('Gain = 1')  # Set the title for the first subplot

plt.subplot(1,2,2)  # Add a subplot in position 2 of a 1x2 grid
plt.imshow(spike_data_sample2.mean(axis=0).reshape((28,-1)).cpu(), cmap='binary')  # Show the mean spike activity for sample 2 as a 28x28 image
plt.axis('off')  # Hide axis ticks and labels
plt.title('Gain = 0.25')  # Set the title for the second subplot

plt.show()  # Display the figure with both subplots
# The above code shows how to visualize the mean spike activity for two different gain values (1 and 0.25) using matplotlib.

# Reshape the sample into a 2D tensor
# Reshape
spike_data_sample2 = spike_data_sample2.reshape((num_steps, -1)) # This will flatten the 28x28 image into a 784-dimensional vector for each time step

# raster plot
fig = plt.figure(facecolor="w", figsize=(10, 5)) # Create a new figure with a white background and a specific size
ax = fig.add_subplot(111) # Add a single subplot to the figure
splt.raster(spike_data_sample2, ax, s=1.5, c="black") # Create a raster plot of the spike data sample with specified point size and color

plt.title("Input Layer") 
plt.xlabel("Time step")
plt.ylabel("Neuron Number")
plt.show() # Show the plot

idx = 210  # index into 210th neuron

fig = plt.figure(facecolor="w", figsize=(8, 1))
ax = fig.add_subplot(111)

splt.raster(spike_data_sample.reshape(num_steps, -1)[:, idx].unsqueeze(1), ax, s=100, c="black", marker="|")
# This will plot the spikes of the 210th neuron over time
plt.title("Input Neuron") 
plt.xlabel("Time step")
plt.yticks([])
plt.show()

def convert_to_time(data, tau=5, threshold=0.01): # Function to convert input data to spike times
  spike_time = tau * torch.log(data / (data - threshold))
  return spike_time

raw_input = torch.arange(0, 5, 0.05) # tensor from 0 to 5
spike_times = convert_to_time(raw_input) # Convert the raw input to spike times using the convert_to_time function

plt.plot(raw_input, spike_times)
plt.xlabel('Input Value')
plt.ylabel('Spike Time (s)')
plt.show()

spike_data = spikegen.latency(data_it, num_steps=100, tau=5, threshold=0.01) # Convert the data to spikes using the latency coding method

fig = plt.figure(facecolor="w", figsize=(10, 5))
ax = fig.add_subplot(111)
splt.raster(spike_data[:, 0].view(num_steps, -1), ax, s=25, c="black") # Create a raster plot of the spike data with specified point size and color

plt.title("Input Layer")
plt.xlabel("Time step")
plt.ylabel("Neuron Number")
plt.show()

fig.savefig('destination_path.png', format='png', dpi=300)

#Linearize the spike times by setting linear=True
spike_data = spikegen.latency(data_it, num_steps=100, tau=5, threshold=0.01, linear=True)

fig = plt.figure(facecolor="w", figsize=(10, 5))
ax = fig.add_subplot(111)
splt.raster(spike_data[:, 0].view(num_steps, -1), ax, s=25, c="black")
plt.title("Input Layer")
plt.xlabel("Time step")
plt.ylabel("Neuron Number")
plt.show()

# Normalize the spike times by setting normalize=True
spike_data = spikegen.latency(data_it, num_steps=100, tau=5, threshold=0.01, normalize=True, linear=True) # Convert the data to spikes using the latency coding method with normalization and linearization

fig = plt.figure(facecolor="w", figsize=(10, 5))
ax = fig.add_subplot(111)
splt.raster(spike_data[:, 0].view(num_steps, -1), ax, s=25, c="black")

plt.title("Input Layer")
plt.xlabel("Time step")
plt.ylabel("Neuron Number")
plt.show()

# Clip the spike times by setting clip=True
spike_data = spikegen.latency(data_it, num_steps=100, tau=5, threshold=0.01, clip=True, normalize=True, linear=True)

fig = plt.figure(facecolor="w", figsize=(10, 5))
ax = fig.add_subplot(111)
splt.raster(spike_data[:, 0].view(num_steps, -1), ax, s=25, c="black")

plt.title("Input Layer")
plt.xlabel("Time step")
plt.ylabel("Neuron Number")
plt.show()

# Create an animation of the mnist latency data
spike_data_sample = spike_data[:, 0, 0]  # Get the sample
print(f"Sample spike data shape: {spike_data_sample.size()}")  # Should be [num_steps, 28, 28]
torch.Size([num_steps, 28, 28])  # Plot the sample using the spike plot module
fig, ax = plt.subplots()  # Create a figure and axis for plotting
anim = splt.animator(spike_data_sample, fig, ax)  # Create the animator for the sample
plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'  # Set the path to ffmpeg for saving the animation
HTML(anim.to_html5_video())  # Display the animation as HTML5 video
anim.save("spike_mnist_latency.mp4")  # Save the animation as an mp4 file
print(f"The corresponding target is: {targets_it[0]}")  # Print the target for the sample