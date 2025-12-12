# Training Output Guide

## Understanding Training Output

When you run `python scripts/train.py`, here's what to expect:

### Initial Setup Output

```
============================================================
Rainbow DQN Training on Space Invaders
============================================================

Creating environment...
State shape: (84, 84, 4)
Number of actions: 6
Device: cpu
⚠️  WARNING: Training on CPU. This will be SLOW!
   For faster training, use a machine with CUDA-capable GPU.

Creating Rainbow DQN agent...
Creating prioritized replay buffer...
Setting up logging...

============================================================
Starting training...
============================================================
```

### During Training

**Clean, minimal output** - One line printed only when evaluation happens (every 50 episodes):

```
Episode 50 | Steps: 523 | Score: 145.0 | Total Steps: 26170 | Mean(100): 132.4 | Eval: 156.3
Episode 100 | Steps: 531 | Score: 180.0 | Total Steps: 52890 | Mean(100): 144.3 | Eval: 178.2
Episode 150 | Steps: 498 | Score: 195.0 | Total Steps: 78450 | Mean(100): 165.8 | Eval: 203.5
```

**Metrics Explained:**
- `Episode`: Current episode number
- `Steps`: Number of steps in this episode
- `Score`: Return for this episode
- `Total Steps`: Total environment steps taken so far
- `Mean(100)`: Average return over last 100 episodes
- `Eval`: Evaluation score (average of 5 test episodes)

### Best Model Tracking

```
New best model saved! Eval: 234.5
```

When evaluation score improves, the best model is automatically saved.

### Checkpointing (Every 100 Episodes)

```
Model saved to checkpoints/rainbow_space_invaders_ep100.pth
```

Regular checkpoints are saved so you can resume training if interrupted.

### Training Complete

```
============================================================
Training completed!
Final model saved to: checkpoints/rainbow_space_invaders_final.pth
Best model saved to: checkpoints/rainbow_space_invaders_best.pth
Logs saved to: logs/rainbow_space_invaders_20251210_120000.csv
============================================================
```

## What Gets Logged

### CSV Log File (`logs/rainbow_space_invaders_*.csv`)

Every episode is logged with:
- episode
- total_steps
- episode_return
- episode_length
- mean_return_10
- mean_return_100
- avg_loss
- buffer_size

Use `python scripts/plot_results.py` to visualize this data.

### TensorBoard (Optional)

If TensorBoard is enabled, real-time metrics are available:

```bash
tensorboard --logdir logs/tensorboard
```

Then open http://localhost:6006

### Checkpoints Saved

- `rainbow_space_invaders_best.pth` - Best performing model (based on evaluation)
- `rainbow_space_invaders_final.pth` - Final model after training
- `rainbow_space_invaders_ep{N}.pth` - Checkpoints every 100 episodes

## Example Training Session

```bash
# Start training
python scripts/train.py

# In another terminal, monitor logs
tail -f logs/rainbow_space_invaders_*.csv

# Or use TensorBoard
tensorboard --logdir logs/tensorboard
```

## Troubleshooting

**Training is too slow on CPU?**
- Reduce `num_episodes` in config (try 50-100 instead of 1000)
- Increase `train_frequency` (train less often)
- Reduce `batch_size`

**Want more frequent updates?**
- Change `episode % 10 == 0` to `episode % 5 == 0` in `scripts/train.py`

**Out of memory?**
- Reduce `batch_size` in config
- Reduce `capacity` of replay buffer
