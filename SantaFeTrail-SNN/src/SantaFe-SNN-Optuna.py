import optuna
import torch
import gymnasium as gym
from SantaFeTrailEnv import SantaFeTrailEnv
from SantaFe_SNN import train_snn

def objective(trial):
    # Suggest hyperparameters
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    gamma = trial.suggest_uniform('gamma', 0.90, 0.999)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    epsilon_decay = trial.suggest_uniform('epsilon_decay', 0.990, 0.9999)
    num_steps = trial.suggest_int('num_steps', 10, 50)
    hidden_size = trial.suggest_categorical('hidden_size', [64, 128, 256, 512])

    # Register and create environment
    try:
        gym.register(
            id="gymnasium_env/SantaFeTrail-v0",
            entry_point="SantaFeTrailEnv:SantaFeTrailEnv",
            reward_threshold=89,
            max_episode_steps=100
        )
    except gym.error.Error:
        pass  # Already registered

    env = gym.make("gymnasium_env/SantaFeTrail-v0")

    # Patch train_snn to accept hidden_size and num_steps
    rewards, _ = train_snn(
        env,
        num_episodes=50,  # Keep small for tuning speed
        batch_size=batch_size,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=epsilon_decay,
        num_steps=num_steps
    )

    avg_reward = sum(rewards[-10:]) / min(10, len(rewards))
    return avg_reward

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")