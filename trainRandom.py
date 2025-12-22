from AtariWrapper import make_atari_env
import os
from datetime import datetime, timedelta
import csv
import numpy as np
from collections import deque
from configs.DQNSpaceInvadersConfig import (
    TRAINING_CONFIG,
    LOGGING_CONFIG,
    ENV_CONFIG
)
import torch

checkpoint_path = LOGGING_CONFIG['random_checkpoint_dir']
logs_path = LOGGING_CONFIG['random_checkpoint_dir']
num_episodes = TRAINING_CONFIG['num_episodes']

from agents.RandomAgent import randomAgent

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train():
    
    env = make_atari_env(ENV_CONFIG['env_id'], render_mode=None)
    agent = randomAgent(env, device)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    os.makedirs(logs_path, exist_ok=True)
    
    log_file = os.path.join(logs_path, f'dqn_space_invaders_{timestamp}.csv')

    rewards = deque(maxlen=100)
    rewards_10 = deque(maxlen=10)

    total_steps = 0
    
    import time
    training_start_time = time.time()

    print(f"Starting training for {TRAINING_CONFIG['num_episodes']} episodes...")
    print(f"Device: {device}")
    print(f"Logging to: {log_file}")

    for episode in range(1, num_episodes+1):
        _, _ = env.reset()

        episode_reward = 0
        episode_steps = 0
        episode_actions = []
        for _ in range(TRAINING_CONFIG['max_steps_per_episode']):
            action = agent.get_action()
            observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_actions.append(action)
            episode_steps += 1
            total_steps += 1


            if done:
                break

        # reward for 10 and 100 episodes
        rewards.append(episode_reward)
        rewards_10.append(episode_reward)

        mean_return_10 = np.mean(list(rewards_10)[-10:]) if len(rewards_10) >= 10 else np.mean(rewards)
        mean_return_100 = np.mean(list(rewards)[-100:]) if len(rewards) >= 100 else np.mean(rewards)

        # Log to CSV every 10 episodes
        if episode % 10 == 0:
            actions_str = str(episode_actions)
            with open(log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([episode, total_steps, episode_reward, episode_steps,
                               mean_return_10, mean_return_100, 0, 0, actions_str])
                
        elapsed_time = time.time() - training_start_time
        time_str = str(timedelta(seconds=int(elapsed_time)))
        
        if episode % LOGGING_CONFIG['console_freq'] == 0:
            print(f"Episode {episode}/{TRAINING_CONFIG['num_episodes']} | "
                  f"Steps: {episode_steps} | "
                  f"Reward: {episode_reward:.2f} | "
                  f"Mean(100): {mean_return_100:.1f} | "
                  f"Total Steps: {total_steps} | "
                  f"Avg Loss: {0:.4f} | "
                  f"Buffer: {0} | "
                  f"Time: {time_str}")
            
        
if __name__ == '__main__':
    print('Staring random agent run')
    train()