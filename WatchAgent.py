import sys
import torch
import numpy as np
import time
from wrappers.AtariWrapper import make_atari_env
from agents.RainbowAgent import RainbowDQN
from configs import ENV_CONFIG, AGENT_CONFIG, DEVICE


def watch_agent(checkpoint_path, num_episodes=5, render_delay=0.02, difficulty=0, mode=0):
    # Load checkpoint first to get the config
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    if 'config' in checkpoint:
        agent_config = checkpoint['config']
        print(f"Using config from checkpoint (num_atoms={agent_config.get('num_atoms', 'N/A')})")
    else:
        print("Warning: No config found in checkpoint, inferring from model structure...")
        num_atoms = checkpoint['online_net']['2.atoms'].shape[0]
        print(f"Detected num_atoms={num_atoms} from checkpoint")
        
        agent_config = AGENT_CONFIG.copy()
        agent_config['num_atoms'] = num_atoms

    print(f"Creating environment with difficulty={difficulty}, mode={mode}")
    env = make_atari_env(
        env_id=ENV_CONFIG['env_id'],
        render_mode='human',  # This enables the visual display
        difficulty=difficulty,
        mode=mode
    )
    
    # Create agent with checkpoint config
    agent = RainbowDQN(config=agent_config)
    
    # Load the weights
    agent.load(checkpoint_path)
    agent.eval_mode()  # Set to evaluation mode (no noise)
    
    print(f"\nWatching agent play {num_episodes} episode(s)...")
    print("Press Ctrl+C to stop\n")
    
    episode_rewards = []
    
    try:
        for episode in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            done = False
            step = 0
            
            while not done:
                state_tensor = torch.FloatTensor(state).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
                action = agent.get_action(state_tensor)
                
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                state = next_state
                step += 1
                
                time.sleep(render_delay)
            
            episode_rewards.append(episode_reward)
            print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Steps = {step}")
    
    except KeyboardInterrupt:
        print("\nWatching interrupted")
    
    finally:
        env.close()
        
        if episode_rewards:
            print(f"\n{'='*50}")
            print(f"Average Reward: {np.mean(episode_rewards):.2f}")
            print(f"Best Reward: {np.max(episode_rewards):.2f}")
            print(f"Worst Reward: {np.min(episode_rewards):.2f}")
            print(f"{'='*50}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python WatchAgent.py <checkpoint_path> [num_episodes] [render_delay] [difficulty] [mode]")
        print("\nArguments:")
        print("  checkpoint_path  : Path to the .pth/.pt checkpoint file")
        print("  num_episodes     : Number of episodes to watch (default: 5)")
        print("  render_delay     : Delay between frames in seconds (default: 0.02)")
        print("  difficulty       : Atari difficulty level 0-1 (default: 0)")
        print("  mode             : Atari game mode 0-15 (default: 0)")
        print("\nExamples:")
        print("  python WatchAgent.py checkpoints/episode_100.pth")
        print("  python WatchAgent.py checkpoints/best_model.pt 10")
        print("  python WatchAgent.py checkpoints/best_model.pt 10 0.02 1 0  # Difficulty 1")
        print("  python WatchAgent.py checkpoints/best_model.pt 10 0.02 0 2  # Mode 2")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    num_episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    render_delay = float(sys.argv[3]) if len(sys.argv) > 3 else 0.02
    difficulty = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    mode = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    
    watch_agent(checkpoint_path, num_episodes, render_delay, difficulty, mode)
