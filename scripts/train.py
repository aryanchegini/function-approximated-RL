"""
Main training script for Rainbow DQN on Space Invaders.
"""
import os
import sys
import random
import numpy as np
import torch
import ale_py
import gymnasium as gym

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.agents.rainbow_agent import RainbowDQNAgent
from src.environment.atari_wrappers import make_atari_env
from src.utils.replay_buffer import PrioritizedReplayBuffer
from src.utils.n_step import NStepBuffer
from src.utils.logger import Logger, TensorBoardLogger
from configs.space_invaders_config import (
    ENV_CONFIG, AGENT_CONFIG, BUFFER_CONFIG, 
    TRAINING_CONFIG, LOGGING_CONFIG, DEVICE, SEED
)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def evaluate_agent(agent, env, num_episodes: int = 5):
    """
    Evaluate agent performance.
    
    Args:
        agent: Rainbow DQN agent
        env: Environment
        num_episodes: Number of episodes to evaluate
    
    Returns:
        Mean episode return
    """
    agent.eval_mode()
    episode_returns = []
    
    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_return = 0
        done = False
        
        while not done:
            # Transpose state from (H, W, C) to (C, H, W)
            state_transposed = np.transpose(state, (2, 0, 1))
            action = agent.select_action(state_transposed)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_return += reward
            state = next_state
        
        episode_returns.append(episode_return)
    
    agent.train_mode()
    return np.mean(episode_returns)


def train():
    """Main training loop."""
    print("="*60)
    print("Rainbow DQN Training on Space Invaders")
    print("="*60)
    
    # Set seed
    set_seed(SEED)
    
    # Register ALE environments
    gym.register_envs(ale_py)
    
    # Create environment
    print("\nCreating environment...")
    env = make_atari_env(ENV_CONFIG['env_id'], render_mode=None)
    eval_env = make_atari_env(ENV_CONFIG['env_id'], render_mode=None)
    
    # Get environment info
    state_shape = env.observation_space.shape  # (84, 84, 4) -> will transpose to (4, 84, 84)
    num_actions = env.action_space.n
    print(f"State shape: {state_shape}")
    print(f"Number of actions: {num_actions}")
    print(f"Device: {DEVICE}")
    if DEVICE == 'cpu':
        print("⚠️  WARNING: Training on CPU. This will be SLOW!")
        print("   For faster training, use a machine with CUDA-capable GPU.")
    else:
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
    
    # Transpose state shape to (C, H, W) for PyTorch
    state_shape_transposed = (state_shape[2], state_shape[0], state_shape[1])
    
    # Create agent
    print("\nCreating Rainbow DQN agent...")
    agent = RainbowDQNAgent(
        state_shape=state_shape_transposed,
        num_actions=num_actions,
        device=DEVICE,
        **AGENT_CONFIG
    )
    
    # Create replay buffer
    print("Creating prioritized replay buffer...")
    replay_buffer = PrioritizedReplayBuffer(
        capacity=BUFFER_CONFIG['capacity'],
        alpha=BUFFER_CONFIG['alpha'],
        beta_start=BUFFER_CONFIG['beta_start'],
        beta_frames=BUFFER_CONFIG['beta_frames']
    )
    
    # Create n-step buffer
    n_step_buffer = NStepBuffer(
        n_step=AGENT_CONFIG['n_step'],
        gamma=AGENT_CONFIG['gamma']
    )
    
    # Create loggers
    print("Setting up logging...")
    logger = Logger(
        log_dir=LOGGING_CONFIG['log_dir'],
        experiment_name='rainbow_space_invaders'
    )
    
    tb_logger = None
    if LOGGING_CONFIG['tensorboard']:
        tb_logger = TensorBoardLogger(
            log_dir=LOGGING_CONFIG['log_dir'],
            experiment_name='rainbow_space_invaders'
        )
    
    # Create checkpoint directory
    os.makedirs(LOGGING_CONFIG['checkpoint_dir'], exist_ok=True)
    
    # Training variables
    total_steps = 0
    episode_returns = []
    best_eval_return = -float('inf')
    
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60 + "\n")
    
    # Training loop
    for episode in range(1, TRAINING_CONFIG['num_episodes'] + 1):
        state, _ = env.reset()
        episode_return = 0
        episode_length = 0
        episode_loss = 0
        num_updates = 0
        
        n_step_buffer.clear()
        done = False
        
        while not done and episode_length < TRAINING_CONFIG['max_steps_per_episode']:
            # Transpose state from (H, W, C) to (C, H, W)
            state_transposed = np.transpose(state, (2, 0, 1))
            
            # Select action
            action = agent.select_action(state_transposed)
            
            # Take step
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_return += reward
            episode_length += 1
            total_steps += 1
            
            # Add to n-step buffer
            next_state_transposed = np.transpose(next_state, (2, 0, 1))
            n_step_result = n_step_buffer.add(
                state_transposed, action, reward, next_state_transposed, done
            )
            
            # Add n-step transition to replay buffer
            if n_step_result is not None:
                s, a, r, ns, d = n_step_result
                replay_buffer.add(s, a, r, ns, d)
            
            # Train agent
            if (total_steps >= TRAINING_CONFIG['learning_starts'] and 
                total_steps % TRAINING_CONFIG['train_frequency'] == 0 and
                len(replay_buffer) >= TRAINING_CONFIG['batch_size']):
                
                batch, indices, weights = replay_buffer.sample(TRAINING_CONFIG['batch_size'])
                loss, priorities = agent.learn(batch, indices, weights)
                replay_buffer.update_priorities(indices, priorities)
                
                episode_loss += loss
                num_updates += 1
            
            state = next_state
        
        episode_returns.append(episode_return)
        avg_loss = episode_loss / num_updates if num_updates > 0 else 0
        
        # Calculate metrics
        mean_return_10 = np.mean(episode_returns[-10:])
        mean_return_100 = np.mean(episode_returns[-100:]) if len(episode_returns) >= 100 else mean_return_10
        
        # Log episode
        metrics = {
            'total_steps': total_steps,
            'episode_return': episode_return,
            'episode_length': episode_length,
            'mean_return_10': mean_return_10,
            'mean_return_100': mean_return_100,
            'avg_loss': avg_loss,
            'buffer_size': len(replay_buffer)
        }
        
        # Always log to CSV (silent)
        logger.log_episode(episode, metrics)
        
        # Log to TensorBoard (silent)
        if tb_logger:
            tb_logger.log_scalar('train/episode_return', episode_return, episode)
            tb_logger.log_scalar('train/mean_return_10', mean_return_10, episode)
            tb_logger.log_scalar('train/mean_return_100', mean_return_100, episode)
            tb_logger.log_scalar('train/episode_length', episode_length, episode)
            tb_logger.log_scalar('train/avg_loss', avg_loss, episode)
        
        # Evaluation - Print summary only here
        if episode % TRAINING_CONFIG['eval_frequency'] == 0:
            eval_return = evaluate_agent(agent, eval_env, TRAINING_CONFIG['eval_episodes'])
            
            # Single clean summary line
            print(f"Episode {episode} | Steps: {episode_length} | Score: {episode_return:.1f} | "
                  f"Total Steps: {total_steps} | Mean(100): {mean_return_100:.1f} | Eval: {eval_return:.1f}")
            
            if tb_logger:
                tb_logger.log_scalar('eval/mean_return', eval_return, episode)
            
            # Save best model
            if eval_return > best_eval_return:
                best_eval_return = eval_return
                best_checkpoint_path = os.path.join(
                    LOGGING_CONFIG['checkpoint_dir'],
                    'rainbow_space_invaders_best.pth'
                )
                agent.save(best_checkpoint_path)
                print(f"New best model saved! Eval return: {eval_return:.2f}")
        
        # Save checkpoint
        if episode % TRAINING_CONFIG['save_frequency'] == 0:
            checkpoint_path = os.path.join(
                LOGGING_CONFIG['checkpoint_dir'],
                f'rainbow_space_invaders_ep{episode}.pth'
            )
            agent.save(checkpoint_path)
    
    # Save final model
    final_checkpoint_path = os.path.join(
        LOGGING_CONFIG['checkpoint_dir'],
        'rainbow_space_invaders_final.pth'
    )
    agent.save(final_checkpoint_path)
    
    # Close loggers
    if tb_logger:
        tb_logger.close()
    
    # Close environments
    env.close()
    eval_env.close()
    
    print("\n" + "="*60)
    print("Training completed!")
    print(f"Final model saved to: {final_checkpoint_path}")
    print(f"Best model saved to: {best_checkpoint_path}")
    print(f"Logs saved to: {logger.get_csv_path()}")
    print("="*60)


if __name__ == '__main__':
    train()
