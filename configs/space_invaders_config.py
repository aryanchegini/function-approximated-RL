"""
Configuration for Rainbow DQN training on Space Invaders.
All hyperparameters in one place for easy experimentation.
"""

# Environment settings
ENV_CONFIG = {
    'env_id': 'ALE/SpaceInvaders-v5',
    'frame_stack': 4,  # Number of frames to stack
    'image_size': 84,  # Preprocessed image size (84x84)
}

# Agent/Network settings
AGENT_CONFIG = {
    'num_atoms': 51,  # Number of atoms for distributional RL (C51)
    'v_min': -10.0,  # Minimum value for distribution support
    'v_max': 10.0,   # Maximum value for distribution support
    'learning_rate': 6.25e-5,  # Learning rate (Adam)
    'gamma': 0.99,  # Discount factor
    'n_step': 3,  # N-step returns
    'target_update_freq': 1000,  # Target network update frequency (steps)
}

# Replay buffer settings
BUFFER_CONFIG = {
    'capacity': 100000,  # Maximum buffer size
    'alpha': 0.6,  # Prioritization exponent (0 = uniform, 1 = full prioritization)
    'beta_start': 0.4,  # Initial importance sampling weight
    'beta_frames': 100000,  # Frames to anneal beta to 1.0
}

# Training settings
TRAINING_CONFIG = {
    'batch_size': 32,  # Batch size for training
    'num_episodes': 1000,  # Number of episodes to train
    'max_steps_per_episode': 10000,  # Maximum steps per episode
    'learning_starts': 10000,  # Start learning after this many steps
    'train_frequency': 4,  # Train every N steps
    'eval_frequency': 10,  # Evaluate every N episodes
    'eval_episodes': 5,  # Number of episodes for evaluation
    'save_frequency': 100,  # Save checkpoint every N episodes
}

# Logging settings
LOGGING_CONFIG = {
    'log_dir': 'logs',
    'checkpoint_dir': 'checkpoints',
    'tensorboard': True,  # Use TensorBoard logging
    'csv_logging': True,  # Use CSV logging
}

# Device settings - automatically use CPU if CUDA is not available
try:
    import torch
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
except ImportError:
    DEVICE = 'cpu'

# Random seed for reproducibility
SEED = 42
