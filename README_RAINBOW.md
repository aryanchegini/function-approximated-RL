# Rainbow DQN Implementation for Space Invaders

A complete, from-scratch implementation of Rainbow DQN (Deep Q-Network with all improvements) trained on Atari Space Invaders. This project features clean architecture, comprehensive logging, and easy-to-use scripts for training and evaluation.

## 🌈 What is Rainbow DQN?

Rainbow DQN combines seven extensions to the original DQN algorithm:

1. **Double Q-Learning** - Reduces overestimation of Q-values
2. **Prioritized Experience Replay** - Samples important transitions more frequently
3. **Dueling Networks** - Separate value and advantage streams
4. **Multi-step Learning** - Uses n-step returns for faster learning
5. **Distributional RL (C51)** - Learns distribution of returns instead of expected value
6. **Noisy Networks** - Learned exploration without epsilon-greedy
7. **Target Network** - Stabilizes training with periodic updates

## 📁 Project Structure

```
RLCoursework/
├── src/                          # Source code
│   ├── agents/                   # Agent implementations
│   │   └── rainbow_agent.py      # Rainbow DQN agent
│   ├── networks/                 # Neural network architectures
│   │   ├── noisy_linear.py       # Noisy linear layer
│   │   └── rainbow_network.py    # Rainbow DQN network
│   ├── environment/              # Environment wrappers
│   │   └── atari_wrappers.py     # Atari preprocessing
│   └── utils/                    # Utility functions
│       ├── replay_buffer.py      # Prioritized replay buffer
│       ├── n_step.py             # N-step return calculator
│       └── logger.py             # Logging utilities
├── configs/                      # Configuration files
│   └── space_invaders_config.py  # Hyperparameters
├── scripts/                      # Executable scripts
│   ├── train.py                  # Training script
│   ├── evaluate.py               # Evaluation script
│   ├── watch_agent.py            # Watch agent play
│   └── plot_results.py           # Plot training curves
├── checkpoints/                  # Saved models (created during training)
├── logs/                         # Training logs (created during training)
└── requirements.txt              # Python dependencies
```

## 🚀 Getting Started

### 1. Environment Setup

#### Create Virtual Environment
```bash
python3.11 -m venv rl-cw-env
source rl-cw-env/bin/activate  # On Windows: rl-cw-env\Scripts\activate
```

#### Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### Install Atari ROMs
```bash
AutoROM --accept-license
```

### 2. Training the Agent

Train Rainbow DQN on Space Invaders:

```bash
python scripts/train.py
```

**Training Features:**
- Automatic checkpointing every 100 episodes
- Best model saved based on evaluation performance
- CSV logging for all metrics
- Optional TensorBoard logging
- Progress printed to console

**Training Configuration:**
All hyperparameters can be modified in `configs/space_invaders_config.py`:
- Network architecture (atoms, value range)
- Learning parameters (learning rate, discount factor)
- Buffer settings (capacity, prioritization)
- Training duration (episodes, steps)

### 3. Watching Your Agent Play

Watch the trained agent play Space Invaders with rendering:

```bash
# Watch best model
python scripts/watch_agent.py --checkpoint checkpoints/rainbow_space_invaders_best.pth --episodes 3

# Watch specific checkpoint
python scripts/watch_agent.py --checkpoint checkpoints/rainbow_space_invaders_ep500.pth --episodes 5
```

### 4. Evaluating Performance

Evaluate agent performance over many episodes (without rendering for speed):

```bash
# Evaluate best model over 100 episodes
python scripts/evaluate.py --checkpoint checkpoints/rainbow_space_invaders_best.pth --episodes 100
```

This will output comprehensive statistics:
- Mean return ± standard deviation
- Median, min, and max returns
- Mean episode length

### 5. Visualizing Training Progress

Plot training curves from logs:

```bash
# Find your log file
ls logs/

# Plot training progress
python scripts/plot_results.py --csv logs/rainbow_space_invaders_TIMESTAMP.csv

# Save plot to file
python scripts/plot_results.py --csv logs/rainbow_space_invaders_TIMESTAMP.csv --save results.png
```

The plot includes:
- Episode returns (raw and smoothed)
- Episode lengths
- Training loss
- Buffer size and total steps

### 6. TensorBoard Visualization

If TensorBoard logging is enabled (default), you can view real-time training metrics:

```bash
tensorboard --logdir logs/tensorboard
```

Then open http://localhost:6006 in your browser.

## 🎮 Environment Details

**Space Invaders Preprocessing:**
- **Frame stacking**: 4 consecutive frames
- **Grayscale conversion**: RGB → grayscale
- **Frame skipping**: Action repeated for 4 frames
- **Reward clipping**: Rewards clipped to {-1, 0, +1}
- **Image size**: 84×84 pixels
- **Normalization**: Pixel values scaled to [0, 1]

## 🧠 Network Architecture

**Input:** 4×84×84 (stacked grayscale frames)

**Convolutional Layers:**
- Conv1: 32 filters, 8×8 kernel, stride 4
- Conv2: 64 filters, 4×4 kernel, stride 2
- Conv3: 64 filters, 3×3 kernel, stride 1

**Dueling Streams:**
- **Value Stream:** 512 hidden units → 51 atoms
- **Advantage Stream:** 512 hidden units → (num_actions × 51 atoms)

**Output:** Distribution over 51 atoms for each action

**Special Features:**
- Noisy linear layers for exploration
- Distributional RL (C51) for learning value distributions

## ⚙️ Hyperparameters

Key hyperparameters (default values in `configs/space_invaders_config.py`):

| Parameter | Value | Description |
|-----------|-------|-------------|
| Learning Rate | 6.25e-5 | Adam optimizer learning rate |
| Discount Factor (γ) | 0.99 | Reward discount |
| N-step | 3 | Steps for n-step returns |
| Target Update | 1000 steps | Target network update frequency |
| Batch Size | 32 | Training batch size |
| Buffer Size | 100,000 | Replay buffer capacity |
| Alpha | 0.6 | Prioritization exponent |
| Beta | 0.4 → 1.0 | Importance sampling weight |
| Num Atoms | 51 | Distribution atoms (C51) |
| V_min / V_max | -10 / +10 | Value distribution support |

## 📊 Expected Performance

With default hyperparameters, you should see:
- **Training time**: ~8-12 hours on GPU (GTX 1080 or better)
- **Initial returns**: 100-200
- **After 500 episodes**: 400-600
- **After 1000 episodes**: 600-1000+
- **Human-level performance**: ~1000-1500

Performance may vary based on random seed and hardware.

## 🔧 Customization

### Modifying Hyperparameters

Edit `configs/space_invaders_config.py` to change any hyperparameters:

```python
AGENT_CONFIG = {
    'learning_rate': 1e-4,  # Increase learning rate
    'gamma': 0.995,         # Higher discount factor
    'n_step': 5,            # More steps for n-step returns
}
```

### Training on Different Atari Games

Modify `ENV_CONFIG` in the config file:

```python
ENV_CONFIG = {
    'env_id': 'ALE/Breakout-v5',  # Or any other Atari game
    'frame_stack': 4,
    'image_size': 84,
}
```

### Adjusting Network Architecture

Modify `RainbowDQNNetwork` in `src/networks/rainbow_network.py` to change:
- Number of convolutional layers
- Filter sizes and counts
- Hidden layer sizes
- Number of atoms for C51

## 🐛 Troubleshooting

### CUDA Out of Memory
- Reduce batch size in config
- Reduce buffer capacity
- Use CPU instead: set `DEVICE = 'cpu'` in config

### Slow Training
- Ensure CUDA is available: `torch.cuda.is_available()`
- Reduce logging frequency
- Disable TensorBoard if not needed

### ROM Not Found
```bash
AutoROM --accept-license
```

### Import Errors
Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

## 📚 References

**Original Papers:**
1. [Rainbow DQN](https://arxiv.org/abs/1710.02298) - Hessel et al., 2017
2. [DQN](https://arxiv.org/abs/1312.5602) - Mnih et al., 2013
3. [Double Q-Learning](https://arxiv.org/abs/1509.06461) - van Hasselt et al., 2015
4. [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952) - Schaul et al., 2015
5. [Dueling DQN](https://arxiv.org/abs/1511.06581) - Wang et al., 2015
6. [Noisy Networks](https://arxiv.org/abs/1706.10295) - Fortunato et al., 2017
7. [Distributional RL](https://arxiv.org/abs/1707.06887) - Bellemare et al., 2017

## 📝 License

This project is for educational purposes as part of University of Bath Reinforcement Learning coursework.

## 🙏 Acknowledgments

- OpenAI Gymnasium for Atari environments
- PyTorch team for the deep learning framework
- Original Rainbow DQN authors

---

**Happy Training! 🎮🤖**
