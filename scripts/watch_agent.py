"""
Watch trained Rainbow DQN agent play Space Invaders.
Loads the best model and renders gameplay.
"""
import os
import sys
import argparse
import numpy as np
import torch
import ale_py
import gymnasium as gym

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.agents.rainbow_agent import RainbowDQNAgent
from src.environment.atari_wrappers import make_atari_env
from configs.space_invaders_config import ENV_CONFIG, AGENT_CONFIG, DEVICE


def watch_agent(checkpoint_path: str, num_episodes: int = 3, render_mode: str = 'human'):
    """
    Watch agent play.
    
    Args:
        checkpoint_path: Path to model checkpoint
        num_episodes: Number of episodes to watch
        render_mode: Rendering mode ('human' or 'rgb_array')
    """
    print("="*60)
    print("Rainbow DQN Agent - Watch Mode")
    print("="*60)
    
    # Register ALE environments
    gym.register_envs(ale_py)
    
    # Create environment with rendering
    print(f"\nCreating environment with render_mode='{render_mode}'...")
    env = make_atari_env(ENV_CONFIG['env_id'], render_mode=render_mode)
    
    # Get environment info
    state_shape = env.observation_space.shape
    num_actions = env.action_space.n
    state_shape_transposed = (state_shape[2], state_shape[0], state_shape[1])
    
    print(f"State shape: {state_shape}")
    print(f"Number of actions: {num_actions}")
    
    # Create agent
    print(f"\nLoading agent from {checkpoint_path}...")
    agent = RainbowDQNAgent(
        state_shape=state_shape_transposed,
        num_actions=num_actions,
        device=DEVICE,
        **AGENT_CONFIG
    )
    
    # Load checkpoint
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return
    
    agent.load(checkpoint_path)
    agent.eval_mode()
    
    print(f"\nWatching agent for {num_episodes} episodes...")
    print("="*60 + "\n")
    
    episode_returns = []
    
    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        episode_return = 0
        episode_length = 0
        done = False
        
        while not done:
            # Transpose state from (H, W, C) to (C, H, W)
            state_transposed = np.transpose(state, (2, 0, 1))
            
            # Select action (greedy)
            action = agent.select_action(state_transposed)
            
            # Take step
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_return += reward
            episode_length += 1
            state = next_state
        
        episode_returns.append(episode_return)
        print(f"Episode {episode}/{num_episodes} - Return: {episode_return:.2f}, Length: {episode_length}")
    
    mean_return = np.mean(episode_returns)
    std_return = np.std(episode_returns)
    
    print("\n" + "="*60)
    print(f"Mean Return: {mean_return:.2f} ± {std_return:.2f}")
    print("="*60)
    
    env.close()


def main():
    parser = argparse.ArgumentParser(description='Watch trained Rainbow DQN agent')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints/rainbow_space_invaders_best.pth',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=3,
        help='Number of episodes to watch'
    )
    parser.add_argument(
        '--render-mode',
        type=str,
        default='human',
        choices=['human', 'rgb_array'],
        help='Rendering mode'
    )
    
    args = parser.parse_args()
    
    watch_agent(
        checkpoint_path=args.checkpoint,
        num_episodes=args.episodes,
        render_mode=args.render_mode
    )


if __name__ == '__main__':
    main()
