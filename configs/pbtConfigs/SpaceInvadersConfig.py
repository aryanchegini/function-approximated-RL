"""
Configuration for Rainbow DQN training on Space Invaders.
"""

# Environment settings
ENV_CONFIG = {
    'env_id': 'ALE/SpaceInvaders-v5',
    'frame_stack': 4,  # Number of frames to stack
    'image_size': 84,  # Preprocessed image size (84x84)
}

PBT_CONFIG = {
    'population_size': 5,
    'exploit_fraction': 0.3,
    'perturb_fraction': 0.2,
}

PBT_AGENTS_CONFIG_TYPE = {
        'learning_rate': 'float',
        'gamma': 'float',
        'batch_size': 'int',
        'per_alpha': 'float',
        'v_min': 'int',
        'v_max': 'int',
        'sigma': 'float',
        'target_update_freq': 'int',
        'num_atoms': 'int',
        'alpha': 'float',
        'beta_start': 'float',
    }

PBT_AGENTS_CONFIG = {
    'lower_bounds':{
        'learning_rate': 1e-5,
        'gamma': 0.95,
        'batch_size': 32, 
        'per_alpha': 0.4,
        'v_min':-100,
        'v_max':10,
        'sigma': 0.1,
        'target_update_freq': 500,
        'num_atoms': 51,
        'alpha': 0.4,
        'beta_start': 0.4,
    },
    'upper_bounds':{
        'learning_rate': 1e-3,
        'gamma': 0.999,
        'batch_size': 128,
        'per_alpha': 0.8,
        'v_min':-10,
        'v_max':100,
        'sigma': 0.5,
        'target_update_freq': 5000,
        'num_atoms': 101,
        'alpha': 0.8,
        'beta_start': 0.8,
    }
}

# Agent/Network settings
STABLE_AGENT_CONFIG = {
    'input_channels': 4,  # Number of stacked frames (matches frame_stack)
    'num_actions': 6,  # Number of actions for Space Invaders
    # 'num_atoms': 51,  # Number of atoms for distributional RL (C51)
    # 'v_min': -10.0,  # Minimum value for distribution support
    # 'v_max': 10.0,   # Maximum value for distribution support
    # 'learning_rate': 6.25e-5,  # Learning rate (Adam)
    # 'gamma': 0.99,  # Discount factor
    'n_step': 3,  # N-step returns
    # 'target_update_freq': 1000,  # Target network update frequency (steps)
}

# Default replay buffer settings
BUFFER_CONFIG = {
    'capacity': 100_000,  # Maximum buffer size
    'alpha': 0.6,  # Prioritization exponent (0 = uniform, 1 = full prioritization)
    'beta_start': 0.4,  # Initial importance sampling weight
    'beta_frames': 100000,  # Frames to anneal beta to 1.0
    'eps': 1e-6,  # Small constant to prevent zero priority
    'seed': None,  # Random seed for buffer sampling (None = random, or use SEED for reproducibility)
}

# Training settings
TRAINING_CONFIG = {
    # 'batch_size': 32,  # Batch size for training
    'num_episodes': 100_000,  # Number of episodes to train estimate is 100 : 1, steps:episodes
    'max_steps_per_episode': 120_000,  # Maximum steps per episode
    'total_training_steps': 1000_000_000,  # Total steps for entire training run
    'learning_starts': 1000,  # Start learning after this many steps
    'train_frequency': 4,  # Train every N steps
    'eval_frequency': 40,  # Evaluate every N episodes
    'eval_episodes': 10,  # Number of episodes for evaluation
    'save_frequency': 5_000,  # Save checkpoint every N episodes
    'eval_seed': 50,  # Seed for evaluation environment
    'change_seed_every': 40  # Change evaluation seed every N evaluations
}

# Logging settings
LOGGING_CONFIG = {
    'log_dir': 'logs',
    'checkpoint_dir': 'pbt_checkpoints',
    'alt_checkpoint_dir': 'bpt_checkpoints_and_logs',
    'tensorboard': True,  # Use TensorBoard logging
    'csv_logging': True,  # Use CSV logging
    'episode_log_frequency': 10,  # Log episodes every N episodes
    'save_global_best': True,  # Save global best model
    'save_periodic_checkpoints': True,  # Save periodic checkpoints
    'num_checkpoints': 20,  # Number of periodic checkpoints
    'checkpoint_by': 'episodes',  # 'steps' or 'episodes'
    'console_every':5
}

# Device settings - automatically use CPU if CUDA is not available
try:
    import torch
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
except ImportError:
    DEVICE = 'cpu'

# Random seed for reproducibility
SEED = 42
