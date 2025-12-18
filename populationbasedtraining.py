from buffers import replay_buffer
import torch
import numpy as np
from numpy.random import uniform, randint
from collections import deque
import time
from datetime import timedelta
import os
import csv
from RainbowAgent import RainbowDQN
from AtariWrapper import make_atari_env
from buffers.replay_buffer import PrioritisedReplayBuffer
from buffers.n_step_buffer import NStepBuffer
import copy

from configs.pbtConfigs.SpaceInvadersConfig import (
    ENV_CONFIG,
    PBT_AGENTS_CONFIG_TYPE,
    STABLE_AGENT_CONFIG,
    BUFFER_CONFIG,
    TRAINING_CONFIG,
    LOGGING_CONFIG,
    DEVICE,
    SEED,
    PBT_CONFIG,
    PBT_AGENTS_CONFIG
)

from scripts.evaluation import eval as evaluate_agent

import multiprocessing as mp
from torch.multiprocessing import Process, Queue, Manager

chekcpoint_path = LOGGING_CONFIG['checkpoint_dir']
logs_path = LOGGING_CONFIG['checkpoint_dir']
eval_seed = TRAINING_CONFIG['eval_seed']

population_size = PBT_CONFIG['population_size']
exploit_fraction = PBT_CONFIG['exploit_fraction']
peturb_fraction = PBT_CONFIG['perturb_fraction']

def rank_members(members, agent_id=None):
    ranked = sorted(members.items(), key=lambda item: item[1]['score'], reverse=True)
    if agent_id is not None:
        agent_rank = next((i for i, (mid, _) in enumerate(ranked) if mid == agent_id), -1)
        return ranked, agent_rank
    return ranked

class Member():
    def __init__(self, agent_config =PBT_AGENTS_CONFIG, unstable_types = PBT_AGENTS_CONFIG_TYPE, stable_config = STABLE_AGENT_CONFIG):
        self.score = -float('inf')  # Initialize with worst fitness
        self.config = {}

        self.agent_config = agent_config
        self.unstable_types = unstable_types
        
        for key in agent_config['lower_bounds'].keys():

            if unstable_types[key]=='int': 
                self.config[key] = randint(
                    agent_config['lower_bounds'][key],
                    agent_config['upper_bounds'][key]+1
                )
            elif unstable_types[key]=='float':
                self.config[key] = uniform(
                    agent_config['lower_bounds'][key],
                    agent_config['upper_bounds'][key]
                )

        for key in stable_config.keys():
            self.config[key] = stable_config[key]

        self.replay_buffer = PrioritisedReplayBuffer(capacity=BUFFER_CONFIG['capacity'], alpha=BUFFER_CONFIG['alpha'], beta_start=BUFFER_CONFIG['beta_start'], beta_frames=BUFFER_CONFIG['beta_frames'], eps=BUFFER_CONFIG['eps'], seed=BUFFER_CONFIG['seed'] )
        self.n_step_buffer = NStepBuffer(n_step=self.config['n_step'], gamma=self.config['gamma'])
        self.agent = RainbowDQN(self.config)

    def evaluate(self, eval_env, seed, episodes=5):
        eval_reward, eval_actions, eval_rewards_eval, eval_states = evaluate_agent(self.agent, eval_env, seed, episodes)
        self.score = eval_reward
        return self.score, eval_actions, eval_rewards_eval, eval_states

    def exploit(self, better_member):
        self.config = copy.deepcopy(better_member.config)
        self.agent = RainbowDQN(self.config)
        self.agent.load_state_dict(better_member.agent.state_dict())
        self.replay_buffer = copy.deepcopy(better_member.replay_buffer)
        self.n_step_buffer = copy.deepcopy(better_member.n_step_buffer)

    def explore(self, unstable_types, agent_config):

        if unstable_types is None:
            unstable_types = self.unstable_types
        
        if agent_config is None:
            agent_config = self.agent_config
        
        for key in unstable_types.keys():
            if uniform(0, 1) < peturb_fraction:
                if unstable_types[key]=='int': 
                    self.config[key] = randint(
                        agent_config['lower_bounds'][key],
                        agent_config['upper_bounds'][key]+1
                    )
                elif unstable_types[key]=='float':
                    self.config[key] = uniform(
                        agent_config['lower_bounds'][key],
                        agent_config['upper_bounds'][key]
                    )
        
        self.agent = RainbowDQN(self.config)
        self.n_step_buffer.gamma = self.config['gamma']


def train_member(id, shared_dict, members):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    agent_path = os.path.join(chekcpoint_path, f'member_{id}_checkpoint.pth')
    log_file = os.path.join(logs_path, f'member_{id}_log.csv')
    eval_log_file = os.path.join(logs_path, f'member_{id}_evaluation_log.csv')

    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'total_steps', 'episode_return', 'episode_length', 
                        'mean_return_10', 'mean_return_100', 'avg_loss', 'buffer_size', 'actions_taken'])
        
    total_steps = 0

    training_start_time = time.time()

    print(f"Starting training for member {id} for {TRAINING_CONFIG['num_episodes']} episodes...")
    print(f"Agent {id} Device: {DEVICE}")
    print(f" Agent {id} Logging to: {log_file}")

    losses = []
    rewards = []
    rewards_deque = deque(maxlen=100)
    best_mean_reward = -float('inf')

    env = make_atari_env(ENV_CONFIG['env_id'], render_mode=None)
    member = Member()
    members[id] = {'member':member, 'score': member.score}
    member.agent.train_mode()


    for episode in range(1, TRAINING_CONFIG['num_episodes'] + 1):
        state, info = env.reset()
        episode_reward = 0
        episode_steps = 0
        episode_actions = []
        member.n_step_buffer.clear()

        episodic_loss = 0 
        batch_sizes = member.config['batch_size'] 
        batch_losses = np.zeros(batch_sizes)


        for _ in range(TRAINING_CONFIG['max_steps_per_episode']):
            state_tensor = torch.FloatTensor(state).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

            member.agent.reset_noise()  # Reset noise for NoisyNets

            action = member.agent.get_action(state_tensor)
            episode_actions.append(action)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_steps += 1

            n_step_transition = member.n_step_buffer.add(state, action, reward, next_state, done)
            
            if n_step_transition:
                s, a, r, s_next, d = n_step_transition
                member.replay_buffer.add(s, a, r, s_next, d)
            state = next_state
            total_steps += 1

            # Training step
            if total_steps > TRAINING_CONFIG['learning_starts'] and total_steps % TRAINING_CONFIG['train_frequency'] == 0:
                if len(member.replay_buffer) >= member.config['batch_size']:
                    batch, indices, weights = member.replay_buffer.sample(member.config['batch_size'])
                    
                    # Convert batch (list of Experience) to dict of tensors
                    batch_dict = {
                        'states': torch.FloatTensor(np.array([exp.state for exp in batch])).permute(0, 3, 1, 2),
                        'actions': torch.LongTensor([exp.action for exp in batch]),
                        'rewards': torch.FloatTensor([exp.reward for exp in batch]),
                        'next_states': torch.FloatTensor(np.array([exp.next_state for exp in batch])).permute(0, 3, 1, 2),
                        'dones': torch.FloatTensor([exp.done for exp in batch]),
                        'weights': torch.FloatTensor(weights)
                    }
                    
                    loss, new_priorities = member.agent.learn(batch_dict, indices, weights)
                    member.replay_buffer.update_priorities(indices, new_priorities)
                    episodic_loss += loss

                    batch_losses = np.append(np.delete(batch_losses, 0), loss)
                    
            if done:
                member.n_step_buffer.clear()
                break

        # Track metrics
        rewards.append(episode_reward)
        rewards_deque.append(episode_reward)
        losses.append(episodic_loss)
        
        # Calculate moving averages
        mean_return_10 = np.mean(list(rewards)[-10:]) if len(rewards) >= 10 else np.mean(rewards)
        mean_return_100 = np.mean(list(rewards_deque)) if len(rewards_deque) > 0 else episode_reward
        avg_loss = np.mean(batch_losses) if np.any(batch_losses > 0) else 0.0
        
        # Log to CSV every 10 episodes
        if episode % 100 == 0:
            actions_str = str(episode_actions)
            with open(log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([episode, total_steps, episode_reward, episode_steps,
                               mean_return_10, mean_return_100, avg_loss, len(member.replay_buffer), actions_str])
        
        # Console logging
        elapsed_time = time.time() - training_start_time
        time_str = str(timedelta(seconds=int(elapsed_time)))
        
        if episode % 100 == 0:
            print(f" Agent {member.id} |  Episode {episode}/{TRAINING_CONFIG['num_episodes']} | "
                  f"Steps: {episode_steps} | "
                  f"Reward: {episode_reward:.2f} | "
                  f"Mean(100): {mean_return_100:.1f} | "
                  f"Total Steps: {total_steps} | "
                  f"Avg Loss: {avg_loss:.4f} | "
                  f"Buffer: {len(member.replay_buffer)} | "
                  f"Time: {time_str}")
            
        if episode % TRAINING_CONFIG['eval_frequency'] == 0:

            reward, _, _, _ = member.evaluate(env, shared_dict['eval_seed'], TRAINING_CONFIG['eval_episodes'])
            shared_dict['eval_count'] += 1

            members[id]['score'] = reward

            print(f" Agent {member.id} Eval after {episode} episodes, {total_steps} steps: Average Reward over {TRAINING_CONFIG['eval_episodes']} episodes: {reward:.2f}, seed:{shared_dict} \n")


            ranked_members, agent_rank = rank_members(members, id)

            if agent_rank >= int(population_size * (1 - exploit_fraction)):
                better_id, better_data = ranked_members[randint(0, int(population_size * (1 - exploit_fraction)))]
                print(f" Agent {member.id} Exploiting member {better_id} with score {better_data['score']:.2f}")
                member.exploit(better_data['member'])
                member.explore()
                print(f" Agent {member.id} New config: {member.config}")

            if shared_dict['eval_count'] % TRAINING_CONFIG['change_seed_every'] == 0:
                shared_dict['eval_seed'] += 1
                print(f" Agent {member.id} Changing eval seed to {shared_dict['eval_seed']}")
                        
        
        # Save best model
        if mean_return_100 > best_mean_reward:
            best_mean_reward = mean_return_100
            best_path = f"{agent_path}/best_model.pt"
            member.agent.save(best_path)
            print(f"New best model saved! Mean reward: {best_mean_reward:.2f}")
        

        # if episode % TRAINING_CONFIG['save_frequency'] == 0 and episode > 0:
        #     ep_checkpoint_path = f"{agent_path}/checkpoint_ep{episode}.pt"
        #     member.agent.save(ep_checkpoint_path)
        #     print(f"Saved checkpoint: {ep_checkpoint_path}") 
  

if __name__ == "__main__":
    with Manager() as manager:
        shared_dict = manager.dict()
        members = manager.dict()

        shared_dict['eval_seed'] = np.random.randint(0, 10000)
        shared_dict['eval_count'] = 0
        
        processes = []

        for i in range(population_size):
            p = Process(target=train_member, args=(i, shared_dict, members))
            p.start()
            processes.append(p)
        
        # Wait for all processes to complete
        for p in processes:
            p.join()