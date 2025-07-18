import torch
import json
import datetime
import os

def save_stats(stats, directory="Data/SantaFeTrail-Generic", prefix="episode_stats"):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(directory, exist_ok=True)
    filename = os.path.join(directory, f"{prefix}_{timestamp}.json")
    with open(filename, "w") as f:
        json.dump(stats, f)

def save_best_transitions(transitions, filename="best_transitions.pt"):
    try:
        processed_transitions = []
        for state, action, reward, next_state, done in transitions:
            processed_transitions.append((
                state.cpu().numpy() if torch.is_tensor(state) else state,
                action,
                reward,
                next_state.cpu().numpy() if torch.is_tensor(next_state) else next_state,
                done
            ))
        torch.save(processed_transitions, filename)
        print(f"Successfully saved {len(processed_transitions)} transitions to {filename}")
    except Exception as e:
        print(f"Error saving transitions: {e}")

def load_best_transitions(filename="best_transitions.pt"):
    try:
        return torch.load(filename, weights_only=False)
    except Exception as e:
        print(f"Error loading transitions: {e}")
        return []

def save_model(model, optimizer, episode_stats, epsilon, filename="model.pt"):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'episode_stats': episode_stats,
        'epsilon': epsilon
    }, filename)

def load_model(model, optimizer, epsilon_start, filename="model.pt"):
    try:
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        episode_stats = checkpoint['episode_stats']
        epsilon = checkpoint['epsilon']
        print(f"Loaded model from {filename} with {len(episode_stats)} episodes")
        return True, episode_stats, epsilon
    except FileNotFoundError:
        print(f"No previous model found at {filename}, starting fresh")
        return False, [], epsilon_start
    except Exception as e:
        print(f"Error loading model: {e}")
        return False, [], epsilon_start
