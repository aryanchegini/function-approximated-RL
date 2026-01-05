import torch
import numpy as np
from collections import deque
import time
from datetime import timedelta
from datetime import datetime
import os
import csv
from agents.RainbowAgent import RainbowDQN
from wrappers.AtariWrapper import make_atari_env
from buffers.replay_buffer import PrioritisedReplayBuffer
from buffers.n_step_buffer import NStepBuffer
from configs.SpaceInvadersConfig import (
    ENV_CONFIG, 
    AGENT_CONFIG, 
    BUFFER_CONFIG, 
    TRAINING_CONFIG,
    LOGGING_CONFIG,
    DEVICE,
    SEED
)

from scripts.evaluation import eval as evaluate_agent

checkpoint_path = LOGGING_CONFIG['checkpoint_dir']
logs_path = LOGGING_CONFIG['checkpoint_dir']

os.makedirs(logs_path, exist_ok=True)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

log_file = os.path.join(logs_path, f'rainbow_space_invaders_{timestamp}.csv')
eval_log_file = os.path.join(logs_path, f'evaluation_{timestamp}.csv')

def create_batch_dict(batch, weights):
    return {
            'states': torch.FloatTensor(np.array([exp.state for exp in batch])).permute(0, 3, 1, 2),
            'actions': torch.LongTensor([exp.action for exp in batch]),
            'rewards': torch.FloatTensor([exp.reward for exp in batch]),
            'next_states': torch.FloatTensor(np.array([exp.next_state for exp in batch])).permute(0, 3, 1, 2),
            'dones': torch.FloatTensor([exp.done for exp in batch]),
            'weights': torch.FloatTensor(weights)
            }

def console_and_log(episode, 
                total_steps, 
                episode_reward, 
                episode_steps, 
                rewards, 
                rewards_deque, 
                batch_losses,
                len_replay_buffer, 
                episode_actions, 
                training_start_time):


        # Calculate moving averages
    prev_10_mean = np.mean(list(rewards)[-10:]) if len(rewards) >= 10 else np.mean(rewards)
    prev_100_mean = np.mean(list(rewards_deque)) if len(rewards_deque) > 0 else episode_reward
    avg_loss = np.mean(batch_losses) if np.any(batch_losses > 0) else 0.0

    # Log to CSV every 10 episodes
    if episode % 10 == 0:
        actions_str = str(episode_actions)
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([episode, total_steps, episode_reward, episode_steps,
                            prev_10_mean, prev_100_mean, avg_loss, len_replay_buffer, actions_str])
            
    # Console logging
    elapsed_time = time.time() - training_start_time
    time_str = str(timedelta(seconds=int(elapsed_time)))
    
    if episode % 10 == 0:
        print(f"Episode {episode}/{TRAINING_CONFIG['num_episodes']} | "
                f"Steps: {episode_steps} | "
                f"Reward: {episode_reward:.2f} | "
                f"Mean(100): {prev_100_mean:.1f} | "
                f"Total Steps: {total_steps} | "
                f"Avg Loss: {avg_loss:.4f} | "
                f"Buffer: {len_replay_buffer} | "
                f"Time: {time_str}")

    return prev_100_mean

def train():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    env = make_atari_env(ENV_CONFIG['env_id'], render_mode=None)
    agent = RainbowDQN()
    
    replay_buffer = PrioritisedReplayBuffer(config_dict=BUFFER_CONFIG, seed=BUFFER_CONFIG['seed'])
    n_step_buffer = NStepBuffer( n_step=AGENT_CONFIG['n_step'], gamma=AGENT_CONFIG['gamma'] )

    # CSV Logging

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'total_steps', 'episode_return', 'episode_length', 
                        'mean_return_10', 'mean_return_100', 'avg_loss', 'buffer_size', 'actions_taken'])
    
    total_steps = 0

    agent.train_mode()
    
    training_start_time = time.time()

    print(f"Starting training for {TRAINING_CONFIG['num_episodes']} episodes...")
    print(f"Device: {DEVICE}")
    print(f"Logging to: {log_file}")

    losses = []
    rewards = []
    rewards_deque = deque(maxlen=100)
    best_mean_reward = -float('inf')
    eval_count = 0

    for episode in range(1, TRAINING_CONFIG['num_episodes'] + 1):
        state, _ = env.reset()
        episode_reward = 0
        episode_steps = 0
        episode_actions = []
        n_step_buffer.clear()

        episodic_loss = 0
        batch_sizes = TRAINING_CONFIG['batch_size']
        batch_losses = np.zeros(batch_sizes)


        for _ in range(TRAINING_CONFIG['max_steps_per_episode']):
            state_tensor = torch.FloatTensor(state).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

            agent.reset_noise()  # Reset noise for NoisyNets

            action = agent.get_action(state_tensor)
            episode_actions.append(action)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_steps += 1

            n_step_transition = n_step_buffer.add(state, action, reward, next_state, done)
            
            if n_step_transition:
                s, a, r, s_next, d = n_step_transition
                replay_buffer.add(s, a, r, s_next, d)

            state = next_state
            total_steps += 1

            # Training step
            if (total_steps > TRAINING_CONFIG['learning_starts'] 
            and total_steps % TRAINING_CONFIG['train_frequency'] == 0
            and len(replay_buffer) >= TRAINING_CONFIG['batch_size']):
                
                batch, indices, weights = replay_buffer.sample(TRAINING_CONFIG['batch_size'])
                
                # Convert batch (list of Experience) to dict of tensors
                batch_dict = batch_dict(batch, weights)
                
                loss, new_priorities = agent.learn(batch_dict, indices, weights)
                replay_buffer.update_priorities(indices, new_priorities)
                episodic_loss += loss

                batch_losses = np.append(np.delete(batch_losses, 0), loss)
                    
            if done:
                n_step_buffer.clear()
                break

        # Track metrics
        rewards.append(episode_reward)
        rewards_deque.append(episode_reward)
        
        prev_100_mean = console_and_log(episode, 
                                        total_steps, 
                                        episode_reward, 
                                        episode_steps,
                                        episodic_loss, 
                                        rewards, 
                                        rewards_deque, 
                                        losses,
                                        len(replay_buffer),
                                        training_start_time)


        losses.append(episodic_loss)
            
        if episode % TRAINING_CONFIG['eval_frequency'] == 0:
            eval_count += 1
            eval_avg_reward, _, _, _ = evaluate_agent(
                env, agent, SEED, 
                num_episodes=TRAINING_CONFIG['eval_episodes']
            )

            print(f"\n Eval after {episode} episodes, {total_steps} steps: Average Reward over {TRAINING_CONFIG['eval_episodes']} episodes: {eval_avg_reward:.2f} \n")

        
        # Save best model
        if prev_100_mean > best_mean_reward:
            best_mean_reward = prev_100_mean
            best_path = f"{checkpoint_path}/best_model.pt"
            agent.save(best_path)
            print(f"New best model saved! Mean reward: {best_mean_reward:.2f}")
        

        if episode % TRAINING_CONFIG['save_frequency'] == 0 and episode > 0:
            ep_checkpoint_path = f"{checkpoint_path}/checkpoint_ep{episode}.pt"
            agent.save(ep_checkpoint_path)
            print(f"Saved checkpoint: {ep_checkpoint_path}") 

    final_checkpoint_path = f"{checkpoint_path}/checkpoint_ep_final.pt"
    agent.save(final_checkpoint_path)
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Final checkpoint: {final_checkpoint_path}")
    print(f"Training log: {log_file}")
    print(f"Final performance (last 100 eps): {mean_return_100:.2f}")
    print(f"{'='*60}") 


if __name__ == '__main__':
    train()
 