import torch
import numpy as np


def eval(env, model, seed=40, num_episodes: int = 5) -> float:
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

    return avg_reward, actions, rewards, states