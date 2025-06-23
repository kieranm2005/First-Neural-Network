import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import SantaFeTrailEnv as sfte
import matplotlib.pyplot as plt


sfte.register() #register the custom environment
env = gym.make("gymnasium_env/SantaFeTrail-v0")

obs, info = env.reset()
done = False

plt.ion() 

while not done:
    env.render()  # Call render method from env
    action = env.action_space.sample()  # Take random actions for preview
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

env.close()

plt.ioff()   # Turn off interactive mode
plt.show() 