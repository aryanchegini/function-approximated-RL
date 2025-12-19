import sys
from buffers import replay_buffer
import torch
import numpy as np
from numpy.random import uniform, randint
from collections import deque
import time
from datetime import timedelta
import os
from RainbowAgent import RainbowDQN
from AtariWrapper import make_atari_env
from buffers.replay_buffer import PrioritisedReplayBuffer
from buffers.n_step_buffer import NStepBuffer
import copy

from Members import Member

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
from logs.pbt_logger import GlobalCheckpointManager

# import multiprocessing as mp
# from torch.multiprocessing import Process, Queue, Manager

chekcpoint_path = LOGGING_CONFIG['checkpoint_dir']
logs_path = LOGGING_CONFIG['checkpoint_dir']
eval_seed = TRAINING_CONFIG['eval_seed']

population_size = PBT_CONFIG['population_size']
exploit_fraction = PBT_CONFIG['exploit_fraction']
peturb_fraction = PBT_CONFIG['perturb_fraction']
checkpoint_path = os.path.join(chekcpoint_path, f'checkpoint.pth')


def rank_members(members, agent_id=None):

    # print('items',members.items())
    # print('A score:', list(members.items())[0][1]['score'])
    ranked = sorted(members.items(), key=lambda item: item[1]['score'], reverse=True)

    # print('ranked:', ranked)

    if agent_id is not None:
        agent_rank = next((i for i, (mid, _) in enumerate(ranked) if mid == agent_id), -1)
        return ranked, agent_rank
    
    return ranked

def create_batch_dict(batch, weights):
    return {
            'states': torch.FloatTensor(np.array([exp.state for exp in batch])).permute(0, 3, 1, 2),
            'actions': torch.LongTensor([exp.action for exp in batch]),
            'rewards': torch.FloatTensor([exp.reward for exp in batch]),
            'next_states': torch.FloatTensor(np.array([exp.next_state for exp in batch])).permute(0, 3, 1, 2),
            'dones': torch.FloatTensor([exp.done for exp in batch]),
            'weights': torch.FloatTensor(weights)
            }

checkpoint_manager = GlobalCheckpointManager(
        base_dir=chekcpoint_path,
        num_checkpoints=LOGGING_CONFIG.get('num_checkpoints', 10),
        total_steps=TRAINING_CONFIG.get('total_training_steps', 10_000_000),
        checkpoint_by=LOGGING_CONFIG.get('checkpoint_by', 'steps')
    )


def train_sequentially(population_size = PBT_CONFIG['population_size']):

    chekcpoint_path = LOGGING_CONFIG['checkpoint_dir']
    logs_path = LOGGING_CONFIG['checkpoint_dir']
    eval_seed = TRAINING_CONFIG['eval_seed']

    exploit_fraction = PBT_CONFIG['exploit_fraction']
    peturb_fraction = PBT_CONFIG['perturb_fraction']
    checkpoint_path = os.path.join(chekcpoint_path, f'checkpoint.pth')

    # Initializing agents
    n_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {n_gpus}")
    device_id = 0  # Always use first GPU
    device = torch.device(f'cuda:{device_id}' if n_gpus > 0 else 'cpu')
    print(f"Training on device: {device}, visible devices: {n_gpus}")

    print(f"Starting training for {TRAINING_CONFIG['num_episodes']} \n on Device: {device} episodes \n with population size of {population_size} ...")

    torch.manual_seed(SEED)    
    np.random.seed(SEED)

    best_mean_reward = -float('inf')

    members = {}
    enviroments = {}
    states = {}
    eval_count = 0
    total_steps = 0

    training_start_time = time.time()

    for i in range(population_size):


        member = Member(i)

        members[i] = {
            'member': member,
            'score': 0,
            'metrics':{'episodes':0, 'steps':0, 'reward':0, 'loss':0, 
                       'rewards_100': deque(maxlen=100), 'rewards_10':deque(maxlen=10)},
            }
        enviroments[i] = make_atari_env(ENV_CONFIG['env_id'], render_mode=None)
        states[i] = {}
        member.agent.train_mode()


    for episode in range(1, TRAINING_CONFIG['num_episodes'] + 1):
        
        for i in range(population_size):
            env = enviroments[i]
            state, info = env.reset()
            member = members[i]['member']
            
            episode_reward = 0
            episode_steps = 0
            # episode_actions = []
            episodic_loss = 0

            try:
                member.n_step_buffer.clear()
                batch_sizes = member.config['batch_size'] 
                batch_losses = np.zeros(batch_sizes)
            except:
                print(f"MEmber {i} config: {member.config}")

            device = member.device

            for _ in range(TRAINING_CONFIG['max_steps_per_episode']):
                state_tensor = torch.FloatTensor(state).permute(2, 0, 1).unsqueeze(0).to(device)

                member.agent.reset_noise()  # Reset noise for NoisyNets

                # Get action from agent and step environment
                action = member.agent.get_action(state_tensor)
                next_state, reward, terminated, truncated, info = env.step(action)


                done = terminated or truncated
                episode_reward += reward
                episode_steps += 1
                total_steps += 1

                
                n_step_transition = member.n_step_buffer.add(state, action, reward, next_state, done)
                
                if n_step_transition:
                    s, a, r, s_next, d = n_step_transition
                    member.replay_buffer.add(s, a, r, s_next, d)
                state = next_state

                # Training step
                if total_steps > TRAINING_CONFIG['learning_starts'] and total_steps % TRAINING_CONFIG['train_frequency'] == 0:
                    if len(member.replay_buffer) >= member.config['batch_size']:
                        batch, indices, weights = member.replay_buffer.sample(member.config['batch_size'])
                        
                        loss, new_priorities = member.agent.learn(create_batch_dict(batch, weights), indices, weights)
                        member.replay_buffer.update_priorities(indices, new_priorities)

                        episodic_loss += loss
                        batch_losses = np.append(np.delete(batch_losses, 0), loss)
                        
                if done:
                    member.n_step_buffer.clear()
                    break

            rewards_100 = members[i]['metrics']['rewards_100']
            rewards_10 = members[i]['metrics']['rewards_10']

            rewards_10.append(episode_reward)
            rewards_100.append(episode_reward)

            # Calculate moving averages
            # mean_return_10 = np.mean(list(rewards)[-10:]) if len(rewards) >= 10 else np.mean(rewards)
            mean_return_10 = np.mean(list(rewards_10)) if len(rewards_10) > 0 else episode_reward
            mean_return_100 = np.mean(list(rewards_100)) if len(rewards_100) > 0 else episode_reward
            avg_loss = np.mean(batch_losses) if np.any(batch_losses > 0) else 0.0

            if episode % LOGGING_CONFIG['episode_log_frequency'] == 0:
                member.logger.log_episode(
                    episode=episode,
                    total_steps=total_steps,
                    episode_return=episode_reward,
                    episode_length=episode_steps,
                    mean_return_10=mean_return_10,
                    mean_return_100=mean_return_100,
                    avg_loss=avg_loss,
                    buffer_size=len(member.replay_buffer)
                )

                        # Console logging
            elapsed_time = time.time() - training_start_time
            time_str = str(timedelta(seconds=int(elapsed_time)))

            if episode % LOGGING_CONFIG['console_every']== 0:
                member.logger.log_console(
                    episode=episode,
                    max_episodes=TRAINING_CONFIG['num_episodes'],
                    episode_steps=episode_steps,
                    episode_reward=episode_reward,
                    mean_return_100=mean_return_100,
                    total_steps=total_steps,
                    avg_loss=avg_loss,
                    buffer_size=len(member.replay_buffer),
                    time_str=time_str
                )

            if episode % TRAINING_CONFIG['eval_frequency'] == 0:
                eval_count += 1
                reward, _, _, _ = member.evaluate(env, eval_seed, TRAINING_CONFIG['eval_episodes'])
                

                members[i]['score'] = reward

                print(f" Agent {member.id} Eval after {episode} episodes, {total_steps} steps: Average Reward over {TRAINING_CONFIG['eval_episodes']} episodes: {reward:.2f}, seed:{eval_seed} \n")

                ranked_members, agent_rank = rank_members(members, i)
                
                if agent_rank >= int(population_size * (1 - exploit_fraction)):

                    # print('ranks:',agent_rank)

                    better_id, better_data = ranked_members[randint(0, int(population_size * (1 - exploit_fraction)))]
                    print(f" Agent {member.id} Exploiting member {better_id} with score {better_data['score']:.2f}")
                    print(f"old config: {member.config}")


                    member.exploit(better_data['member'], episode=episode, total_steps=total_steps)
                    member.explore(episode=episode, total_steps=total_steps)
                

                if eval_count % TRAINING_CONFIG['change_seed_every'] == 0:
                    eval_seed += 1
                    print(f" Agent {member.id} Changing eval seed to {eval_count}")

                if agent_rank == 0:
                    print(f" Agent {member.id} is the best member!")

                    # Update global best
                    if checkpoint_manager.update_best(
                        member_id=member.id,
                        episode=episode,
                        total_steps=total_steps,
                        score=mean_return_100,
                        agent=member.agent,
                        config=member.config
                    ):
                        best_mean_reward = reward
                        print(f" Agent {member.id} New global best model saved! Score: {reward:.2f}")

                if checkpoint_manager.should_save_checkpoint(total_steps):
                    best_member_id = ranked_members[0][0]
                    best_member = members[best_member_id]['member']
                    checkpoint_manager.save_checkpoint(
                        member_id=best_member_id,
                        episode=episode,
                        total_steps=total_steps,
                        score=mean_return_100,
                        agent=best_member.agent,
                        config=best_member.config
                    )
                    print(f" Saved periodic checkpoint at {total_steps} steps (best: member {best_member_id})")
        


if __name__ == "__main__":

    population_size = int(sys.argv[1]) if len(sys.argv) > 1 else PBT_CONFIG['population_size']

    train_sequentially(population_size)