# Rainbow DQN for Atari Space Invaders

A complete, from-scratch implementation of Rainbow DQN (Deep Q-Network with all 7 improvements) for training on Atari games.

## 🌈 What's Implemented

This project includes a **complete Rainbow DQN implementation** with:

1. **Double Q-Learning** - Reduces overestimation
2. **Prioritized Experience Replay** - Samples important transitions
3. **Dueling Networks** - Separate value and advantage streams
4. **Multi-step Learning** - N-step returns
5. **Distributional RL (C51)** - Learns return distribution
6. **Noisy Networks** - Learned exploration
7. **Target Network** - Stabilizes training

## 📚 Documentation

- **[README_RAINBOW.md](README_RAINBOW.md)** - Complete usage guide and documentation
- **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** - Implementation overview
- **[PIPELINE_OVERVIEW.md](PIPELINE_OVERVIEW.md)** - Architecture diagrams
- **[FILE_STRUCTURE.md](FILE_STRUCTURE.md)** - Explanation of all files

## 🚀 Quick Start

### 1. Setup
```bash
# Activate virtual environment
source rl-cw-env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install Atari ROMs
AutoROM --accept-license
```

### 2. Test Installation
```bash
python test_installation.py
```

### 3. Train Agent
```bash
python scripts/train.py
```

### 4. Watch Agent Play
```bash
python scripts/watch_agent.py --checkpoint checkpoints/rainbow_space_invaders_best.pth
```

### 5. Evaluate Performance
```bash
python scripts/evaluate.py --checkpoint checkpoints/rainbow_space_invaders_best.pth
```

### 6. Plot Results
```bash
python scripts/plot_results.py --csv logs/rainbow_space_invaders_*.csv
```

## 📁 Project Structure

```
RLCoursework/
├── src/              # Source code (all from scratch)
│   ├── agents/       # Rainbow DQN agent
│   ├── networks/     # Network architecture
│   ├── environment/  # Atari wrappers
│   └── utils/       # Replay buffer, logging
├── configs/         # Hyperparameters
├── scripts/         # Train, evaluate, watch
├── checkpoints/     # Saved models
└── logs/           # Training logs
```

## 📖 For Complete Documentation

See **[README_RAINBOW.md](README_RAINBOW.md)** for comprehensive documentation including:
- Detailed setup instructions
- Hyperparameter explanations
- Customization guide
- Troubleshooting
- References to original papers
