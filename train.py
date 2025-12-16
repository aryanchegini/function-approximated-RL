import torch
import numpy as np
from collections import deque
from RainbowAgent import RainbowDQN
from AtariWrapper import make_atari_env
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


def train():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    env = make_atari_env(ENV_CONFIG['env_id'], render_mode=None)
    agent = RainbowDQN()
    
    replay_buffer = PrioritisedReplayBuffer( capacity=BUFFER_CONFIG['capacity'], alpha=BUFFER_CONFIG['alpha'], beta_start=BUFFER_CONFIG['beta_start'], beta_frames=BUFFER_CONFIG['beta_frames'], eps=BUFFER_CONFIG['eps'], seed=BUFFER_CONFIG['seed'] )
    
    n_step_buffer = NStepBuffer( n_step=AGENT_CONFIG['n_step'], gamma=AGENT_CONFIG['gamma'] )

    total_steps = 0
    agent.train_mode()

    print(f"Starting training for {TRAINING_CONFIG['num_episodes']} episodes...")
    print(f"Device: {DEVICE}")

# check Architecture size params
# make notebook which saves best weights, displays training curves, videos of agent playing etc.
# run in vast or hex. 
# compare with other rainbow implementations.

    losses = []
    rewards = deque(maxlen=100)

    for episode in range(TRAINING_CONFIG['num_episodes']):
        state, info = env.reset()
        episode_reward = 0
        episode_steps = 0
        n_step_buffer.clear()  # Clear n-step buffer at episode start

        episodic_loss = 0
        batch_sizes = TRAINING_CONFIG['batch_size']
        batch_losses = np.zeros(batch_sizes)


        for _ in range(TRAINING_CONFIG['max_steps_per_episode']):
            state_tensor = torch.FloatTensor(state).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

            agent.reset_noise()  # Reset noise for NoisyNets

            action = agent.get_action(state_tensor)
            next_state, reward, terminated, truncated, info = env.step(action)
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
            if total_steps > TRAINING_CONFIG['learning_starts'] and total_steps % TRAINING_CONFIG['train_frequency'] == 0:
                if len(replay_buffer) >= TRAINING_CONFIG['batch_size']:
                    batch, indices, weights = replay_buffer.sample(TRAINING_CONFIG['batch_size'])
                    
                    # Convert batch (list of Experience) to dict of tensors
                    batch_dict = {
                        'states': torch.FloatTensor(np.array([exp.state for exp in batch])).permute(0, 3, 1, 2),
                        'actions': torch.LongTensor([exp.action for exp in batch]),
                        'rewards': torch.FloatTensor([exp.reward for exp in batch]),
                        'next_states': torch.FloatTensor(np.array([exp.next_state for exp in batch])).permute(0, 3, 1, 2),
                        'dones': torch.FloatTensor([exp.done for exp in batch]),
                        'weights': torch.FloatTensor(weights)
                    }
                    
                    loss, new_priorities = agent.learn(batch_dict, indices, weights)
                    replay_buffer.update_priorities(indices, new_priorities)
                    episodic_loss += loss

                    batch_losses = np.append(np.delete(batch_losses, 0), loss)
                    rewards.append(episode_reward)
                    


            if done:
                n_step_buffer.clear()
                break

        # Logging
        print(f"Episode {episode}/{TRAINING_CONFIG['num_episodes']} | "
              f"Steps: {episode_steps} | "
              f"Reward: {episode_reward:.2f} | "
              f"R100 {np.mean(rewards):.1f} | "
              f"Total Steps: {total_steps} | "
              f" | Avg Loss: {np.mean(batch_losses):.4f} | "
              f"Episode Loss: {episodic_loss:.4f}"
              )
        rewards.append(episode_reward)
        losses.append(episodic_loss)

        
        # Save checkpoint
        if episode % TRAINING_CONFIG['save_frequency'] == 0 and episode > 0:
            checkpoint_path = f"{LOGGING_CONFIG['checkpoint_dir']}/checkpoint_ep{episode}.pt"
            agent.save(checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}") 


if __name__ == '__main__':
    train()
