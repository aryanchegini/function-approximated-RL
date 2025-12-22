
"""
The configuration for DQN on space invaders
"""


ENV_CONFIG = {
    'env_id': 'ALE/SpaceInvaders-v5',
    'frame_stack': 4,  # Number of frames to stack
    'image_size': 84,  # Preprocessed image size (84x84)
}

AGENT_CONFIG = {
    'input_channels': 4,  # Number of stacked frames (matches frame_stack)
    'num_actions': 6,  # Number of actions for Space Invaders
    'learning_rate': 3e-4,  # Learning rate (Adam)
    'gamma': 0.99,  # Discount factor
    'n_step': 3,  # N-step returns
    'tau': 0.005,  # Target network update frequency (steps) 
    'batch_size': 128,  # Batch size for training
    'eps_decay':2500,
    'eps_start':0.9,
    'eps_end':0.01,
    'buffer_capacity':10_000
}

# Training settings
TRAINING_CONFIG = {
    'batch_size': 128,  # Batch size for training
    'num_episodes': 100_000,  # Number of episodes to train
    'max_steps_per_episode': 10000,  # Maximum steps per episode
    'learning_starts': 1000,  # Start learning after this many steps
    'train_frequency': 4,  # Train every N steps
    'eval_frequency': 1000,  # Evaluate every N episodes
    'eval_episodes': 100,  # Number of episodes for evaluation
    'save_frequency': 200_000,  # Save checkpoint every N episodes
    'eval_seed':50
}

LOGGING_CONFIG = {
    'console_freq': 50,
    'log_dir': 'logs/DQN',
    'checkpoint_dir': 'checkpoints/DQN',
    'alt_checkpoint_dir': 'vast_ai_checkpoints_and_logs/DQN',
    'random_checkpoint_dir': 'checkpoints/Random',
    'tensorboard': True,  # Use TensorBoard logging
    'csv_logging': True,  # Use CSV logging
}