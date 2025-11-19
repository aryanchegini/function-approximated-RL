import gymnasium as gym
import ale_py
import numpy as np
import torch

from agilerl.algorithms.dqn_rainbow import RainbowDQN

ENV_ID = "ALE/SpaceInvaders-v5"
CHECKPOINT_PATH = "models/rainbow_space_invaders.pt" 

gym.register_envs(ale_py)

def to_channels_first(obs: np.ndarray) -> np.ndarray:
    """Convert (N, H, W, C) -> (N, C, H, W)."""
    return np.moveaxis(obs, -1, 1)


def make_agent():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Load RainbowDQN directly from checkpoint.
    # This reconstructs the agent with the same architecture & config.
    agent = RainbowDQN.load(CHECKPOINT_PATH, device=device)
    agent.set_training_mode(False)
    return agent


if __name__ == "__main__":
    # Human-rendering env so you can see the game
    env = gym.make(ENV_ID, render_mode="human")
    obs, info = env.reset()

    agent = make_agent()

    num_episodes = 3

    for ep in range(num_episodes):
        done = False
        truncated = False
        obs, info = env.reset()
        total_reward = 0.0

        while not (done or truncated):
            # Add batch dimension: (1, H, W, C)
            obs_batch = obs[None, ...]
            # Convert to channels-first: (1, C, H, W)
            obs_cf = to_channels_first(obs_batch)

            # Get action from agent (greedy policy)
            action = agent.get_action(obs_cf)

            if isinstance(action, np.ndarray):
                action = int(action[0])
            elif torch.is_tensor(action):
                action = int(action.item())
            else:
                action = int(action)

            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward

        print(f"Episode {ep+1}/{num_episodes} finished with return {total_reward}")

    env.close()
