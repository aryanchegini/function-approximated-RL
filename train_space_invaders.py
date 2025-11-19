import os
import csv
import numpy as np
import torch
import gymnasium as gym
import ale_py

from agilerl.utils.utils import make_vect_envs, observation_space_channels_to_first
from agilerl.components.replay_buffer import ReplayBuffer
from agilerl.components.data import Transition
from agilerl.algorithms.dqn_rainbow import RainbowDQN

# ----------------- CONFIG ----------------- #
ENV_ID = "ALE/SpaceInvaders-v5"
NUM_ENVS = 1                     # keep 1 env for simplicity
TRAIN_EPISODES = 300             # increase for a stronger agent
MAX_STEPS_PER_EPISODE = 2000

LOG_DIR = "logs"
CSV_PATH = os.path.join(LOG_DIR, "rainbow_space_invaders.csv")
CHECKPOINT_DIR = "models"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "rainbow_space_invaders.pt")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# --------------- UTIL FUNCTIONS ----------- #
gym.register_envs(ale_py)


def to_channels_first(obs: np.ndarray) -> np.ndarray:
    """Convert (N, H, W, C) -> (N, C, H, W)."""
    return np.moveaxis(obs, -1, 1)


# --------------- ENV + AGENT SETUP -------- #
env = make_vect_envs(ENV_ID, num_envs=NUM_ENVS, should_async_vector=False)

observation_space = env.single_observation_space
action_space = env.single_action_space

print("Raw observation space:", observation_space)          # (210, 160, 3)
observation_space_cf = observation_space_channels_to_first(observation_space)
print("Channels-first observation space:", observation_space_cf)  # (3, 210, 160)
print("Action space:", action_space)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

NET_CONFIG = {
    "encoder_config": {
        "channel_size": [32, 32],
        "kernel_size": [8, 4],
        "stride_size": [4, 2],
    },
    "head_config": {"hidden_size": [256]},
}

agent = RainbowDQN(
    observation_space=observation_space_cf,
    action_space=action_space,
    net_config=NET_CONFIG,
    batch_size=32,
    lr=1e-4,
    learn_step=4,
    gamma=0.99,
    tau=1e-3,
    n_step=3,
    device=device,
)
agent.set_training_mode(True)

memory = ReplayBuffer(
    max_size=5_000,
    device=device,
)

# --------------- CSV SETUP ----------------- #
# log: episode, total_env_steps, episode_return, episode_length, mean_return_last_10
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["episode", "total_env_steps", "episode_return", "episode_length", "mean_return_last_10"]
        )

# --------------- TRAINING LOOP ------------- #
total_env_steps = 0
returns_history = []

for ep in range(1, TRAIN_EPISODES + 1):
    obs, info = env.reset()               # (num_envs, H, W, C)
    obs_cf = to_channels_first(obs)       # (num_envs, C, H, W)

    done = np.array([False] * NUM_ENVS)
    episode_return = 0.0
    episode_length = 0

    while not done.any() and episode_length < MAX_STEPS_PER_EPISODE:
        # Get action from Rainbow agent
        action = agent.get_action(obs_cf)

        # Ensure action is a numpy array of ints of shape (NUM_ENVS,)
        action = np.array(action).astype(int)

        next_obs, reward, terminated, truncated, infos = env.step(action)
        next_obs_cf = to_channels_first(next_obs)

        done = np.logical_or(terminated, truncated)

        # For NUM_ENVS=1, reward is shape (1,), so take reward[0]
        reward_scalar = float(reward[0])
        episode_return += reward_scalar
        episode_length += 1
        total_env_steps += NUM_ENVS

        # Store transition (channels-first observations)
        transition = Transition(
            obs=obs_cf,
            action=action,
            reward=reward,
            next_obs=next_obs_cf,
            done=done,
            batch_size=[NUM_ENVS],
        )
        memory.add(transition.to_tensordict())

        # Learn from replay buffer once we have enough samples
        if len(memory) >= agent.batch_size:
            updates = max(1, NUM_ENVS // agent.learn_step)
            for _ in range(updates):
                experiences = memory.sample(agent.batch_size)
                agent.learn(experiences)

        obs_cf = next_obs_cf

    returns_history.append(episode_return)
    mean_last_10 = np.mean(returns_history[-10:])

    # Print progress
    print(
        f"Episode {ep}/{TRAIN_EPISODES} | "
        f"Return: {episode_return:.2f} | "
        f"Length: {episode_length} | "
        f"Mean(Last10): {mean_last_10:.2f} | "
        f"TotalSteps: {total_env_steps}"
    )

    # Append to CSV
    with open(CSV_PATH, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [ep, total_env_steps, episode_return, episode_length, mean_last_10]
        )

# --------------- SAVE CHECKPOINT ----------- #
agent.set_training_mode(False)
agent.save_checkpoint(CHECKPOINT_PATH)
print(f"Saved trained Rainbow agent to: {CHECKPOINT_PATH}")
print(f"Logged training metrics to: {CSV_PATH}")

env.close()
