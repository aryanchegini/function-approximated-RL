"""
Evaluate trained Rainbow DQN agent on Space Invaders.
Runs multiple episodes without rendering and reports statistics.
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


def evaluate_agent(checkpoint_path: str, num_episodes: int = 100):
    """
    Evaluate agent performance over multiple episodes.
    
    Args:
        checkpoint_path: Path to model checkpoint
        num_episodes: Number of episodes to evaluate
    """
    print("="*60)
    print("Rainbow DQN Agent - Evaluation Mode")
    print("="*60)
    
    # Register ALE environments
    gym.register_envs(ale_py)
    
    # Create environment (no rendering for faster evaluation)
    print("\nCreating environment...")
    env = make_atari_env(ENV_CONFIG['env_id'], render_mode=None)
    
    # Get environment info
    state_shape = env.observation_space.shape
    num_actions = env.action_space.n
    state_shape_transposed = (state_shape[2], state_shape[0], state_shape[1])
    
    # Create agent
    print(f"Loading agent from {checkpoint_path}...")
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
    
    print(f"\nEvaluating agent for {num_episodes} episodes...")
    print("="*60 + "\n")
    
    episode_returns = []
    episode_lengths = []
    
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
        episode_lengths.append(episode_length)
        
        if episode % 10 == 0:
            print(f"Episode {episode}/{num_episodes} - Return: {episode_return:.2f}, Length: {episode_length}")
    
    # Calculate statistics
    mean_return = np.mean(episode_returns)
    std_return = np.std(episode_returns)
    min_return = np.min(episode_returns)
    max_return = np.max(episode_returns)
    median_return = np.median(episode_returns)
    
    mean_length = np.mean(episode_lengths)
    
    print("\n" + "="*60)
    print("Evaluation Results:")
    print("="*60)
    print(f"Number of episodes: {num_episodes}")
    print(f"Mean return: {mean_return:.2f} ± {std_return:.2f}")
    print(f"Median return: {median_return:.2f}")
    print(f"Min return: {min_return:.2f}")
    print(f"Max return: {max_return:.2f}")
    print(f"Mean episode length: {mean_length:.1f}")
    print("="*60)
    
    env.close()
    
    return {
        'mean_return': mean_return,
        'std_return': std_return,
        'min_return': min_return,
        'max_return': max_return,
        'median_return': median_return,
        'mean_length': mean_length,
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained Rainbow DQN agent')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints/rainbow_space_invaders_best.pth',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=100,
        help='Number of episodes to evaluate'
    )
    
    args = parser.parse_args()
    
    evaluate_agent(
        checkpoint_path=args.checkpoint,
        num_episodes=args.episodes
    )


if __name__ == '__main__':
    main()
