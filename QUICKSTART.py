"""
Quick Start Guide for Rainbow DQN Training
"""

# QUICK START GUIDE
# ==================

# 1. SETUP (One-time)
# --------------------
# Activate virtual environment
# source rl-cw-env/bin/activate

# Install dependencies (if not already done)
# pip install -r requirements.txt

# Install ROMs (if not already done)
# AutoROM --accept-license


# 2. TRAIN AGENT
# ----------------
# python scripts/train.py

# This will:
# - Train Rainbow DQN for 1000 episodes (configurable)
# - Save checkpoints every 100 episodes
# - Save best model based on evaluation
# - Log metrics to CSV and TensorBoard


# 3. WATCH AGENT PLAY
# --------------------
# python scripts/watch_agent.py --checkpoint checkpoints/rainbow_space_invaders_best.pth --episodes 3


# 4. EVALUATE PERFORMANCE
# ------------------------
# python scripts/evaluate.py --checkpoint checkpoints/rainbow_space_invaders_best.pth --episodes 100


# 5. PLOT RESULTS
# ----------------
# python scripts/plot_results.py --csv logs/rainbow_space_invaders_TIMESTAMP.csv --save results.png


# CONFIGURATION
# --------------
# Edit configs/space_invaders_config.py to modify:
# - Training duration (num_episodes)
# - Learning rate, batch size, etc.
# - Network architecture parameters
# - Logging and checkpointing frequency


# TIPS
# -----
# 1. Training takes ~8-12 hours on GPU for 1000 episodes
# 2. You can stop training anytime (Ctrl+C) - progress is saved
# 3. Resume training by loading a checkpoint and continuing
# 4. Monitor with TensorBoard: tensorboard --logdir logs/tensorboard
# 5. Best model is saved automatically based on evaluation


# EXPECTED PERFORMANCE
# ---------------------
# Episode 100:  ~200-300 return
# Episode 500:  ~400-600 return
# Episode 1000: ~600-1000+ return
# Human-level:  ~1000-1500 return
