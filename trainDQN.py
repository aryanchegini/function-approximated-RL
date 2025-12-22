from agents.DQN import DQN
import torch
from AtariWrapper import make_atari_env
import os
from datetime import datetime, timedelta
import csv
import numpy as np
from configs.DQNSpaceInvadersConfig import (
    TRAINING_CONFIG,
    LOGGING_CONFIG,
    ENV_CONFIG
)
from collections import deque
from scripts.evaluation import eval as evaluate_agent


if torch.cuda.is_available() and hasattr(torch, 'set_float32_matmul_precision'):
    # Check if GPU supports TF32 (Ampere or newer - compute capability >= 8.0)
    props = torch.cuda.get_device_properties(0)
    if props.major >= 8:
        torch.set_float32_matmul_precision('high')
        print("Using TF32 for faster training")
    else:
        print(f"GPU {props.name} doesn't support TF32 (compute capability {props.major}.{props.minor})")
        torch.set_float32_matmul_precision('medium')
        print(f'Using torch float32 matmult precision on medium')

checkpoint_path = LOGGING_CONFIG['checkpoint_dir']
logs_path = LOGGING_CONFIG['checkpoint_dir']
seed = TRAINING_CONFIG['eval_seed']
num_episodes = TRAINING_CONFIG['num_episodes']

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train():

   
    agent = DQN(device, env)
        
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    os.makedirs(logs_path, exist_ok=True)

    log_file = os.path.join(logs_path, f'dqn_space_invaders_{timestamp}.csv')
    eval_log_file = os.path.join(logs_path, f'evaluation_{timestamp}.csv')

    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'total_steps', 'episode_return', 'episode_length', 
                        'mean_return_10', 'mean_return_100', 'avg_loss', 'buffer_size', 'actions_taken'])
        
    rewards = deque(maxlen=100)
    rewards_10 = deque(maxlen=10)
    batch_losses = deque(maxlen=agent.batch_size)
    best_mean_reward = -float('inf')
    eval_count = 0

    total_steps = 0


    import time
    training_start_time = time.time()

    print(f"Starting training for {TRAINING_CONFIG['num_episodes']} episodes...")
    print(f"Device: {device}")
    print(f"Logging to: {log_file}")


    for episode in range(1, num_episodes+1):
        # Initialize the environment and get its state
        state, info = env.reset()
        # state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        state_tensor = torch.FloatTensor(state).permute(2, 0, 1).unsqueeze(0).to(device)
        episode_reward = 0
        episode_steps = 0
        episode_actions = []
        for _ in range(TRAINING_CONFIG['max_steps_per_episode']):
            action = agent.get_action(state_tensor)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward_tensor = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.FloatTensor(observation).permute(2, 0, 1).unsqueeze(0).to(device)
                # next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory
            agent.memory.push(state_tensor, action, next_state, reward_tensor)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            loss = agent.optimize_model()

            # logging updates
            episode_reward += reward
            episode_actions.append(action)

            
            if loss is not None:
                batch_losses.append(loss.item() if isinstance(loss, torch.Tensor) else loss)


            if done:
                break
        
        # reward for 10 and 100 episodes
        rewards.append(episode_reward)
        rewards_10.append(episode_reward)

        mean_return_10 = np.mean(list(rewards_10)[-10:]) if len(rewards_10) >= 10 else np.mean(rewards)
        mean_return_100 = np.mean(list(rewards)[-100:]) if len(rewards) >= 100 else np.mean(rewards)
        # avg_loss = np.mean([loss.item() if isinstance(loss, torch.Tensor) else loss for loss in batch_losses]) if len(batch_losses) > 0 else 0
        try:
            avg_loss = np.mean(list(batch_losses)) if len(batch_losses) > 0 else 0
        except:
            print(batch_losses)
        
        # Log to CSV every 10 episodes
        if episode % 10 == 0:
            actions_str = str(episode_actions)
            with open(log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([episode, total_steps, episode_reward, episode_steps,
                               mean_return_10, mean_return_100, avg_loss, agent.memory.__len__(), actions_str])
        
        # Console logging
        elapsed_time = time.time() - training_start_time
        time_str = str(timedelta(seconds=int(elapsed_time)))
        
        if episode % LOGGING_CONFIG['console_freq'] == 0:
            print(f"Episode {episode}/{TRAINING_CONFIG['num_episodes']} | "
                  f"Steps: {episode_steps} | "
                  f"Reward: {episode_reward:.2f} | "
                  f"Mean(100): {mean_return_100:.1f} | "
                  f"Total Steps: {total_steps} | "
                  f"Avg Loss: {avg_loss:.4f} | "
                  f"Buffer: {agent.memory.__len__()} | "
                  f"Time: {time_str}")
            
        if episode % TRAINING_CONFIG['eval_frequency'] == 0:
            eval_count += 1
            eval_avg_reward, eval_actions, eval_rewards_eval, eval_states = evaluate_agent(
                env, agent, seed, 
                num_episodes=TRAINING_CONFIG['eval_episodes'])

            print(f"\n Eval after {episode} episodes, {total_steps} steps: Average Reward over {TRAINING_CONFIG['eval_episodes']} episodes: {eval_avg_reward:.2f} \n")

            
            # Save best model
            if mean_return_100 > best_mean_reward:
                best_mean_reward = mean_return_100
                best_path = f"{checkpoint_path}/best_model.pt"
                agent.save(best_path)
                print(f"New best model saved! Mean reward: {best_mean_reward:.2f}")
            

        if episode % TRAINING_CONFIG['save_frequency'] == 0 and episode > 0:
            ep_checkpoint_path = f"{checkpoint_path}/checkpoint_ep{episode}.pt"
            agent.save(ep_checkpoint_path)
            print(f"Saved checkpoint: {ep_checkpoint_path}") 


if __name__ == '__main__':
    train()
    print('training finished')

