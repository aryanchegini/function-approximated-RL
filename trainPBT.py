import sys
import torch
import numpy as np
from numpy.random import randint
from collections import deque
import time
from datetime import timedelta
from wrappers.AtariWrapper import make_atari_env
import copy

from configs.pbtConfigs.SpaceInvadersConfig import (
    ENV_CONFIG,
    TRAINING_CONFIG,
    LOGGING_CONFIG,
    SEED,
    PBT_CONFIG,
)
from agents.Members import Member
from logs.pbt_logger import GlobalCheckpointManager

import multiprocessing as mp
from torch.multiprocessing import Process, Queue, Manager

from logs.pbt_logger import GlobalCheckpointManager

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

chekcpoint_path = LOGGING_CONFIG['checkpoint_dir']


eval_seed = TRAINING_CONFIG['eval_seed']
exploit_fraction = PBT_CONFIG['exploit_fraction']

checkpoint_manager = GlobalCheckpointManager(
    base_dir=chekcpoint_path,
    num_checkpoints=LOGGING_CONFIG.get('num_checkpoints', 10),
    total_steps=TRAINING_CONFIG.get('total_training_steps', 10_000_000),
    checkpoint_by=LOGGING_CONFIG.get('checkpoint_by', 'steps')
)



def rank_members(shared_dict, agent_id):
    
    ranked = sorted(shared_dict.items(), key = lambda item: item[1]['score'], reverse=True)

    if ranked[0][1]['score'] > 0:
        agent_rank = next((i for i, (mid, _) in enumerate(ranked) if mid == agent_id), -1)
    else:
        agent_rank = 0
    return ranked, agent_rank

def state_dict_to_device(state_dict, device):
    return {k: v.to(device) for k, v in state_dict.items()}

def evaluate(env, i, member, shared_dict, eval_data, episode, total_steps):

    score, _, _, _ = member.evaluate(env, eval_data['eval_seed'], TRAINING_CONFIG['eval_episodes'])

    temp_member = shared_dict[i]
    temp_member['score'] = score
    shared_dict[i] = temp_member

    ranked_members, rank = rank_members(shared_dict, i)
    total_size = len(ranked_members)

    #Checks if the agent's rank is in the bottom 'exploit_fraction' of agents (and score has updated)
    if rank >= int(total_size * (1-exploit_fraction)) and ranked_members[0][1]['score'] > 0:

        better_id =  ranked_members[randint(0, int(total_size * (1 - exploit_fraction)))][0]
        best_score = ranked_members[better_id][1]['score']

        print(f" Agent {member.id} with score {score} exploiting member {better_id} with score {best_score:.2f}")

        better_config = copy.deepcopy(shared_dict[better_id]['configs'])
        better_params = state_dict_to_device(shared_dict[better_id]['state_dict'], device)

        member.exploit(better_config, better_params, better_id, episode=episode)
        member.explore(episode=episode, total_steps=total_steps)

    shared_dict = add_to_shared_dict(i, shared_dict, member.agent.state_dict(), member.config, score, device)

    if eval_data['eval_count'] % TRAINING_CONFIG['change_seed_every'] == 0:
        eval_data['eval_seed'] += randint(10,50)
        print(f"   Agent {member.id} Changing eval seed to {eval_data['eval_count']}")

    return shared_dict, rank, score


def add_to_shared_dict(i, shared_dict, state_dict, config, score, device):

    state = {k: v.cpu() for k, v in state_dict.items()}
    temp = {'configs':config, 'state_dict': state, 'score':score, 'device': device}
    shared_dict[i] = temp

    return shared_dict

def create_batch_dict(batch, weights):
    return {
            'states': torch.FloatTensor(np.array([exp.state for exp in batch])).permute(0, 3, 1, 2),
            'actions': torch.LongTensor([exp.action for exp in batch]),
            'rewards': torch.FloatTensor([exp.reward for exp in batch]),
            'next_states': torch.FloatTensor(np.array([exp.next_state for exp in batch])).permute(0, 3, 1, 2),
            'dones': torch.FloatTensor([exp.done for exp in batch]),
            'weights': torch.FloatTensor(weights)
            }


def update_best(member, checkpoint_manager, episode, mean, steps):
    if checkpoint_manager.update_best(
            member_id=member.id,
            episode=episode,
            total_steps=member.agent.learn_step_counter,
            score=mean,
            agent=member.agent,
            config=member.config
        ):

        print(f" Agent {member.id} New global best model saved! Score: {mean:.2f}")

    if checkpoint_manager.should_save_checkpoint(steps):
        
        checkpoint_manager.save_checkpoint(
            member_id=i,
            episode=episode,
            total_steps=member.agent.learn_step_counter,
            score=mean,
            agent=member.agent,
            config=member.config
        )

        print(f" Saved periodic checkpoint at {steps} steps (best: member {i})")

def training_thread(id, device, thread_population, shared_dict, devices, eval_data):
    torch.manual_seed(SEED + id)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    population_size = len(thread_population)
    
    thread_members = {}
    enviroments = {}
    metrics = {}

    for i in thread_population:
        
        member = Member(i, device)
 
        # shared files can only update one 'layer' at a time
        temp = shared_dict[i]
        temp['configs'] = member.config
        temp['state_dict'] = state_dict_to_device(member.agent.state_dict(), torch.device('cpu'))
        shared_dict[i] = temp

        thread_members[i] = member
        enviroments[i] = make_atari_env(ENV_CONFIG['env_id'], render_mode=None)
        member.agent.train_mode()

        metrics[i] = {'episodes':0, 'reward':0, 'loss':0, 'total_steps':0,
                       'rewards_100': deque(maxlen=100), 'rewards_10':deque(maxlen=10)}

    training_start_time = time.time()
    print(f'Started training thread {id} with {population_size} members ({thread_population}) on device {device}')

    for episode in range(1, TRAINING_CONFIG['num_episodes'] + 1):
        
        for i in thread_population:

            total_steps = metrics[i]['total_steps']
            env = enviroments[i]
            state, _ = env.reset()
            member = thread_members[i]

            episode_reward = 0
            episode_steps = 0

            member.n_step_buffer.clear()
            batch_sizes = member.config['batch_size'] 
            batch_losses = np.zeros(batch_sizes)

            for _ in range(TRAINING_CONFIG['max_steps_per_episode']):
                state_tensor = torch.FloatTensor(state).permute(2, 0, 1).unsqueeze(0).to(device)
                
                member.agent.reset_noise()  # Reset noise for NoisyNets

                # Get action from agent and step environment
                action = member.agent.get_action(state_tensor)
                next_state, reward, terminated, truncated, _ = env.step(action)

                done = terminated or truncated
                episode_reward += reward
                episode_steps += 1
                metrics[i]['total_steps'] += 1
                total_steps += 1

                # add a step to the n step buffer
                n_step_transition = member.n_step_buffer.add(state, action, reward, next_state, done)
                
                # add a step to the prioritized replay buffer
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

                        batch_losses = np.append(np.delete(batch_losses, 0), loss)

                if done:
                    member.n_step_buffer.clear()
                    break

            prev_100 = metrics[i]['rewards_100']
            prev_10 = metrics[i]['rewards_10']

            prev_100.append(episode_reward)
            prev_10.append(episode_reward)

            # Calculate moving averages
            prev_100_mean = np.mean(list(prev_100)) if len(prev_100) > 0 else episode_reward
            prev_10_mean = np.mean(list(prev_10)) if len(prev_10) > 0 else episode_reward
            avg_loss = np.mean(batch_losses) if np.any(batch_losses > 0) else 0.0
            
            if episode % LOGGING_CONFIG['episode_log_frequency'] == 0:
                member.logger.log_episode(
                    episode=episode,
                    total_steps=total_steps,
                    episode_return=episode_reward,
                    episode_length=episode_steps,
                    mean_return_10=prev_10_mean,
                    mean_return_100=prev_100_mean,
                    avg_loss=avg_loss,
                    buffer_size=len(member.replay_buffer)
                )

            elapsed_time = time.time() - training_start_time
            time_str = str(timedelta(seconds=int(elapsed_time)))

            if episode % LOGGING_CONFIG['console_every']== 0:
                member.logger.log_console(
                    episode=episode,
                    max_episodes=TRAINING_CONFIG['num_episodes'],
                    episode_steps=episode_steps,
                    episode_reward=episode_reward,
                    mean_return_100=prev_100_mean,
                    total_steps=total_steps,
                    avg_loss=avg_loss,
                    buffer_size=len(member.replay_buffer),
                    time_str=time_str
                )

            if episode % TRAINING_CONFIG['eval_frequency'] == 0 and total_steps > TRAINING_CONFIG['learning_starts']:
                eval_data['eval_count'] += 1

                shared_dict, rank, score = evaluate(env, i, member, shared_dict, eval_data, episode, total_steps)

                print(f" Agent {member.id} Eval after {episode} episodes, {total_steps} steps: Average Reward over {TRAINING_CONFIG['eval_episodes']} episodes: {score:.2f}, seed:{eval_data['eval_seed']}")


                if rank == 0:
                    # Update global best

                    update_best(member, checkpoint_manager, episode, prev_100_mean, total_steps)


    print(f'Thread {id} on device:{device} finished training after {episode} episodes')


if __name__=='__main__':

    device_count = torch.cuda.device_count()
    population_size = int(sys.argv[1]) if len(sys.argv) > 1 else PBT_CONFIG['population_size']

    print(f"Number of GPUs available: {device_count}")

    cpu = torch.device("cpu")
    gpus = {}
    devices = {}
    
    # Creates the list of gpus
    for device_id in range(torch.cuda.device_count()):
        gpu = torch.device(f'cuda:{device_id}' if device_count > 0 else 'cpu')
        gpus[f'cuda:{device_id}'] = gpu

    
    # Manages console logging, saving models and file logs 
    if len(gpus) == 0:
        # Runs a single cpu thread

        device = cpu

        print(f"Running on CPU only, cpu: {device}")

        shared_dict = {}
        eval_data = {}
        eval_data['eval_seed'] = eval_seed
        eval_data['eval_count'] = 0
        
        devices_dict = { str(device):{'members':[], '.device':device }}
        # A dict (this doesnt change) of devices, the members on a device and device instance

        for i in range(population_size):
            # Allocate each member of population a gpu
            # add a place for an agent, score and device for each member of the population
            
            shared_dict[i] = { 'configs':None, 'state_dict': None, 'score':0, 'device':str(device) }
            devices_dict[str(device)]['members'].append(i)
            print(f'Running agent {i} on :{str(device)}')

        training_thread(0, 
                        device, 
                        devices_dict[str(device)]['members'], 
                        shared_dict,
                        devices, 
                        eval_data)   

    elif len(gpus) >= 1:
        # Runs one or more gpu threads

        mp.set_start_method('spawn', force=True) # Allows multiprocessing
    
        with Manager() as manager:

            shared_dict = manager.dict() 
            eval_data = manager.dict() 
            eval_data['eval_seed'] = eval_seed
            eval_data['eval_count'] = 0
            # A shared dictionary containing a dict of members, 
            # each member has an agent, score and device

            devices_dict = { device:{'members':[], '.device':gpus[str(device)] } for device in gpus.keys() }
            # A dict (this doesnt change) of devices, the members on a device and device instance

            print('Available gpus:')
            for i, device in enumerate(gpus):
                print(f'Device {i}, name:{device}, and instance: {devices_dict[str(device)]}')
            print('\n')

            n_gpus = len(gpus)

            for i in range(population_size):
                # Allocate each member of population a gpu
                device_id = i % n_gpus
                # add a place for an agent, score and device for each member of the population
                shared_dict[i] = {'configs':None, 'state_dict': None, 'score':0, 'device':f'cuda:{device_id}' }
                devices_dict[f'cuda:{device_id}']['members'].append(i)
                print(f"Running agent {i} on gpu:{f'cuda:{device_id}'}")

            processes = []
            # Place each process on respective gpus
            for i, gpu in enumerate(gpus):
                process = Process(target = training_thread,
                    args = (i, 
                        gpus[gpu], 
                        devices_dict[gpu]['members'], 
                        shared_dict,
                        devices, eval_data
                        ))
                
                process.start()
                processes.append(process)
            # Runs each process
            for i, p in enumerate(processes):
                p.join()
                print(f'Thread {i} terminated')

        print('Training Complete')


    




            


