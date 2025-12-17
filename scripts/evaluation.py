import torch
import numpy as np
import csv
import os


def eval(env, model, seed=40, num_episodes: int = 5, eval_count: int = 0, log_file: str = None) -> float:
    """
    Evaluate the model for a number of episodes and return the average reward.
    """
    env.reset(seed=seed)
    model.eval_mode() # Sets RainbowDQN to evaluation mode (no noise from NoisyNets)
    total_reward = 0.0

    actions = []
    rewards = []
    states = []

    for episode in range(num_episodes):

        ep_actions = []
        ep_rewards = []
        ep_states = []

        state, _ = env.reset(seed=seed + episode)
        done = False
        episode_reward = 0.0

        while not done:
            state_tensor = torch.FloatTensor(state).permute(2, 0, 1).unsqueeze(0).to(model.device)
            with torch.no_grad():
                action = model.get_action(state_tensor)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            state = next_state

            ep_actions.append(action)
            ep_rewards.append(reward)
            ep_states.append(state)

        total_reward += episode_reward
        actions.append(ep_actions)
        rewards.append(ep_rewards)  
        states.append(ep_states)

    avg_reward = total_reward / num_episodes

    model.train_mode()  # Sets RainbowDQN back to training mode

    # Save to CSV if log_file provided
    if log_file:
        file_exists = os.path.exists(log_file)
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['eval_number', 'episode', 'actions', 'rewards', 'avg_reward'])
            
            for ep_idx in range(num_episodes):
                writer.writerow([
                    eval_count,
                    ep_idx + 1,
                    str(actions[ep_idx]),
                    str(rewards[ep_idx]),
                    total_reward / num_episodes
                ])

    return avg_reward, actions, rewards, states