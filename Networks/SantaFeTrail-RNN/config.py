# Hyperparameters and config for SantaFeTrail-RNN

num_episodes = 600
batch_size = 64
gamma = 0.9765900240544403 #optuna
epsilon_start = 1.0
epsilon_end = 0.1
epsilon_decay = 0.9621836956639912 #optuna
learning_rate = 0.001471632325723092 #optuna
replay_buffer_size = 50000
recent_buffer_size = 5000
hidden_size = 256 #optuna
target_update_freq = 1000
n_step = 6 #optuna
