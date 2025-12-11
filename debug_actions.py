"""
Debug script to check what actions the agent is taking
"""
import numpy as np
import torch
from collections import Counter
from src.agents.rainbow_agent import RainbowDQNAgent
from src.environment.atari_wrappers import make_atari_env
from configs.space_invaders_config import ENV_CONFIG, AGENT_CONFIG

# Create environment
env = make_atari_env(ENV_CONFIG['env_id'])

# Get action meanings
import gymnasium as gym
base_env = gym.make(ENV_CONFIG['env_id'])
action_meanings = base_env.unwrapped.get_action_meanings()
print(f"Available actions: {action_meanings}")
base_env.close()

# Load agent
state_shape = env.observation_space.shape[0]
num_actions = env.action_space.n

agent = RainbowDQNAgent(
    state_shape=state_shape,
    num_actions=num_actions,
    **AGENT_CONFIG
)

print("\nLoading best model...")
agent.load("checkpoints/rainbow_space_invaders_best.pth")

# Track actions over one episode
print("\nTracking actions for 1 episode...\n")
state, _ = env.reset()
state = np.array(state)
state = np.transpose(state, (2, 0, 1))

done = False
action_counts = Counter()
step = 0

while not done and step < 1000:
    action = agent.select_action(state)
    action_counts[action] += 1
    
    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    
    next_state = np.array(next_state)
    state = np.transpose(next_state, (2, 0, 1))
    step += 1

# Print results
print(f"Episode lasted {step} steps\n")
print("Action Distribution:")
print("-" * 50)
for action_id in sorted(action_counts.keys()):
    count = action_counts[action_id]
    percentage = (count / step) * 100
    action_name = action_meanings[action_id]
    print(f"Action {action_id} ({action_name:>15}): {count:4d} times ({percentage:5.1f}%)")

print("\n" + "=" * 50)
if len(action_counts) <= 2:
    print("⚠️  WARNING: Agent is only using 1-2 actions!")
    print("   This suggests it's stuck in a local optimum.")
    print("   The agent needs more exploration during training.")
else:
    print("✅ Agent is using multiple actions.")

env.close()
