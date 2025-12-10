# Rainbow DQN Project Structure

## 📁 Complete File Structure

```
RLCoursework/
│
├── 📄 README_RAINBOW.md          # Main documentation (comprehensive guide)
├── 📄 PROJECT_SUMMARY.md         # Project overview and what's implemented
├── 📄 PIPELINE_OVERVIEW.md       # Detailed pipeline and architecture diagrams
├── 📄 FILE_STRUCTURE.md          # This file - explains each file
├── 📄 QUICKSTART.py              # Quick reference for common commands
├── 📄 test_installation.py       # Test script to verify setup
├── 📄 requirements.txt           # Python dependencies
├── 📄 .gitignore                 # Git ignore rules
│
├── 📁 src/                       # Source code (all from scratch)
│   ├── 📄 __init__.py
│   │
│   ├── 📁 agents/                # Agent implementations
│   │   ├── 📄 __init__.py
│   │   └── 📄 rainbow_agent.py   # Rainbow DQN agent
│   │       ├── RainbowDQNAgent class
│   │       ├── Double Q-learning
│   │       ├── Distributional RL (C51)
│   │       ├── Target network updates
│   │       ├── Model save/load
│   │       └── Training/eval modes
│   │
│   ├── 📁 networks/              # Neural network architectures
│   │   ├── 📄 __init__.py
│   │   ├── 📄 noisy_linear.py    # Noisy linear layer
│   │   │   ├── NoisyLinear class
│   │   │   ├── Factorized Gaussian noise
│   │   │   ├── Parameter initialization
│   │   │   └── Noise reset mechanism
│   │   │
│   │   └── 📄 rainbow_network.py # Rainbow DQN network
│   │       ├── RainbowDQNNetwork class
│   │       ├── Conv feature extractor (Nature DQN)
│   │       ├── Dueling architecture (value + advantage)
│   │       ├── Noisy layers for exploration
│   │       ├── C51 distributional output (51 atoms)
│   │       └── Q-value computation from distributions
│   │
│   ├── 📁 environment/           # Environment wrappers
│   │   ├── 📄 __init__.py
│   │   └── 📄 atari_wrappers.py  # Atari preprocessing
│   │       ├── NoopResetEnv (random initial states)
│   │       ├── MaxAndSkipEnv (4-frame skip + max pool)
│   │       ├── FireResetEnv (auto-fire at start)
│   │       ├── EpisodicLifeEnv (life loss = episode end)
│   │       ├── ClipRewardEnv (clip to {-1, 0, +1})
│   │       ├── WarpFrame (84x84 grayscale)
│   │       ├── FrameStack (stack 4 frames)
│   │       ├── ScaledFloatFrame (normalize to [0, 1])
│   │       └── make_atari_env() helper function
│   │
│   └── 📁 utils/                 # Utility functions
│       ├── 📄 __init__.py
│       ├── 📄 replay_buffer.py   # Prioritized replay buffer
│       │   ├── SumTree class (efficient prioritized sampling)
│       │   ├── PrioritizedReplayBuffer class
│       │   ├── Add experiences with priorities
│       │   ├── Sample batch (prioritized)
│       │   ├── Update priorities (TD errors)
│       │   └── Beta annealing for importance sampling
│       │
│       ├── 📄 n_step.py           # N-step returns
│       │   ├── NStepBuffer class
│       │   ├── Accumulate n-step rewards
│       │   ├── Handle episode boundaries
│       │   └── Compute n-step returns
│       │
│       └── 📄 logger.py           # Logging utilities
│           ├── Logger class (CSV logging)
│           ├── TensorBoardLogger class
│           ├── Episode metrics logging
│           └── Console output formatting
│
├── 📁 configs/                   # Configuration files
│   └── 📄 space_invaders_config.py  # All hyperparameters
│       ├── ENV_CONFIG (environment settings)
│       ├── AGENT_CONFIG (network & learning params)
│       ├── BUFFER_CONFIG (replay buffer settings)
│       ├── TRAINING_CONFIG (training duration & frequency)
│       ├── LOGGING_CONFIG (log paths & options)
│       ├── DEVICE (cuda/cpu)
│       └── SEED (reproducibility)
│
├── 📁 scripts/                   # Executable scripts
│   ├── 📄 train.py               # Main training script
│   │   ├── Setup environment & agent
│   │   ├── Initialize replay buffer
│   │   ├── Training loop
│   │   ├── Logging (CSV + TensorBoard)
│   │   ├── Periodic evaluation
│   │   ├── Checkpointing (regular + best)
│   │   └── Progress monitoring
│   │
│   ├── 📄 watch_agent.py         # Watch agent play (with rendering)
│   │   ├── Load trained model
│   │   ├── Play N episodes with rendering
│   │   └── Display performance stats
│   │
│   ├── 📄 evaluate.py            # Evaluate agent performance
│   │   ├── Load trained model
│   │   ├── Run N episodes (no rendering)
│   │   ├── Collect statistics
│   │   └── Report mean/std/min/max returns
│   │
│   └── 📄 plot_results.py        # Visualize training progress
│       ├── Load CSV logs
│       ├── Plot episode returns (raw & smoothed)
│       ├── Plot episode lengths
│       ├── Plot training loss
│       ├── Plot buffer size & steps
│       └── Save/display plots
│
├── 📁 checkpoints/               # Saved models (created during training)
│   ├── rainbow_space_invaders_best.pth      # Best performing model
│   ├── rainbow_space_invaders_final.pth     # Final model after training
│   └── rainbow_space_invaders_ep{N}.pth     # Checkpoint every 100 episodes
│
└── 📁 logs/                      # Training logs (created during training)
    ├── rainbow_space_invaders_TIMESTAMP.csv  # CSV metrics log
    └── tensorboard/              # TensorBoard event files
        └── rainbow_space_invaders_TIMESTAMP/
```

## 🔍 Key File Descriptions

### Core Implementation Files (Most Important)

#### `src/networks/rainbow_network.py`
**Purpose**: Complete Rainbow DQN network architecture  
**Key Features**:
- Nature DQN convolutional layers
- Dueling architecture (separate value and advantage streams)
- Distributional RL (outputs distribution over 51 atoms)
- Noisy layers for exploration
- ~2.5M parameters

#### `src/agents/rainbow_agent.py`
**Purpose**: Rainbow DQN agent with full learning algorithm  
**Key Features**:
- Double Q-learning (action selection vs evaluation)
- Categorical projection for distributional Bellman update
- N-step bootstrapping
- Target network management
- Model save/load functionality
- Training/evaluation mode switching

#### `src/utils/replay_buffer.py`
**Purpose**: Prioritized experience replay  
**Key Features**:
- Sum tree data structure (O(log n) operations)
- Priority-based sampling (alpha parameter)
- Importance sampling weights (beta annealing)
- Automatic priority updates based on TD errors

#### `src/networks/noisy_linear.py`
**Purpose**: Noisy networks for exploration  
**Key Features**:
- Factorized Gaussian noise
- Learnable noise parameters (mu and sigma)
- Training vs eval mode (noise on/off)
- Replaces epsilon-greedy exploration

#### `src/environment/atari_wrappers.py`
**Purpose**: Standard Atari preprocessing  
**Key Features**:
- Frame skipping and max pooling
- Grayscale conversion and resizing
- Reward clipping
- Frame stacking
- Episodic life management

### Training & Evaluation Scripts

#### `scripts/train.py`
**Purpose**: Main training loop  
**Usage**: `python scripts/train.py`  
**Features**:
- Complete training pipeline
- Automatic checkpointing
- CSV and TensorBoard logging
- Periodic evaluation
- Best model tracking

#### `scripts/watch_agent.py`
**Purpose**: Visualize trained agent  
**Usage**: `python scripts/watch_agent.py --checkpoint path/to/model.pth --episodes 3`  
**Features**:
- Load any checkpoint
- Render gameplay
- Display episode returns

#### `scripts/evaluate.py`
**Purpose**: Statistical performance evaluation  
**Usage**: `python scripts/evaluate.py --checkpoint path/to/model.pth --episodes 100`  
**Features**:
- Fast evaluation (no rendering)
- Comprehensive statistics
- Mean ± std, median, min/max

#### `scripts/plot_results.py`
**Purpose**: Visualize training progress  
**Usage**: `python scripts/plot_results.py --csv logs/file.csv --save plot.png`  
**Features**:
- 4-panel plot (returns, length, loss, buffer/steps)
- Smoothing for clarity
- Save or display

### Configuration & Documentation

#### `configs/space_invaders_config.py`
**Purpose**: Central configuration  
**Contains**:
- All hyperparameters
- Environment settings
- Training parameters
- Easy to modify for experiments

#### `README_RAINBOW.md`
**Purpose**: Main documentation  
**Contains**:
- Getting started guide
- Complete usage instructions
- Hyperparameter explanations
- Troubleshooting
- References

#### `PROJECT_SUMMARY.md`
**Purpose**: Implementation overview  
**Contains**:
- What's implemented
- Architecture details
- Expected results
- Key features

#### `PIPELINE_OVERVIEW.md`
**Purpose**: Visual pipeline explanation  
**Contains**:
- ASCII diagrams of architecture
- Training loop flow
- Data flow diagrams
- Learning algorithm details

### Utility Files

#### `test_installation.py`
**Purpose**: Verify setup  
**Usage**: `python test_installation.py`  
**Tests**:
- All dependencies installed
- Environment creation
- Custom modules importable
- Agent creation
- Forward pass

#### `requirements.txt`
**Purpose**: Python dependencies  
**Usage**: `pip install -r requirements.txt`  
**Contains**:
- PyTorch
- Gymnasium with Atari
- OpenCV
- Plotting libraries
- TensorBoard

#### `QUICKSTART.py`
**Purpose**: Quick reference  
**Contains**:
- Common commands
- Usage examples
- Configuration tips
- Expected performance

## 🎯 Most Important Files to Understand

For learning Rainbow DQN, study these in order:

1. **`configs/space_invaders_config.py`** - See all hyperparameters
2. **`src/networks/noisy_linear.py`** - Understand noisy networks
3. **`src/networks/rainbow_network.py`** - See full architecture
4. **`src/utils/replay_buffer.py`** - Understand prioritized replay
5. **`src/utils/n_step.py`** - See n-step returns
6. **`src/agents/rainbow_agent.py`** - Complete learning algorithm
7. **`src/environment/atari_wrappers.py`** - Environment preprocessing
8. **`scripts/train.py`** - See it all come together

## 🔧 Files to Modify for Experimentation

- **Change hyperparameters**: `configs/space_invaders_config.py`
- **Modify network**: `src/networks/rainbow_network.py`
- **Adjust training loop**: `scripts/train.py`
- **Try different game**: Change `env_id` in config

## 📊 Output Files (Generated During Training)

- **Checkpoints**: `checkpoints/*.pth` - Saved models
- **CSV logs**: `logs/*.csv` - Training metrics
- **TensorBoard**: `logs/tensorboard/` - Real-time visualization
- **Plots**: Generated by `plot_results.py`

---

**Total Lines of Code**: ~2,500 lines  
**Total Files**: 20+ files  
**Implementation**: 100% from scratch  
**Status**: Production ready ✅
