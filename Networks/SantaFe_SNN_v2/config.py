# Hyperparameters and config for SantaFeTrail-RNN

num_episodes = 300
batch_size = 64
gamma = 0.9150113438723146 #optuna
epsilon_start = 1.0
epsilon_end = 0.1
epsilon_decay = 0.9068292089222714 #optuna
learning_rate = 8.226576718852825e-05 #optuna
replay_buffer_size = 50000
recent_buffer_size = 5000
hidden_size = 85 #optuna
target_update_freq = 100
n_step = 6 #optuna
