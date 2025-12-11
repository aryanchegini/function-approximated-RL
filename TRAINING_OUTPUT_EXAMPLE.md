# Training Output Example

## What You'll See When Training

### Initial Setup
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

### During Training (Silent Until Evaluation)

Training runs silently... episodes are being completed and logged to CSV, but nothing is printed to console.

### At Evaluation Time (Every 50 Episodes)

One clean line per evaluation:

```
Episode 50 | Steps: 1247 | Score: 145.0 | Total Steps: 62350 | Mean(100): 132.4 | Eval: 156.3
Episode 100 | Steps: 1198 | Score: 180.0 | Total Steps: 125400 | Mean(100): 144.3 | Eval: 178.2
Episode 150 | Steps: 1156 | Score: 195.0 | Total Steps: 187650 | Mean(100): 165.8 | Eval: 203.5
Episode 200 | Steps: 1234 | Score: 210.0 | Total Steps: 251200 | Mean(100): 178.2 | Eval: 224.1
New best model saved! Eval: 224.1
Episode 250 | Steps: 1189 | Score: 230.0 | Total Steps: 314750 | Mean(100): 189.5 | Eval: 241.3
New best model saved! Eval: 241.3
Episode 300 | Steps: 1276 | Score: 255.0 | Total Steps: 378900 | Mean(100): 203.7 | Eval: 268.9
New best model saved! Eval: 268.9
```

### When Checkpoints Are Saved (Every 100 Episodes)

```
Episode 100 | Steps: 1198 | Score: 180.0 | Total Steps: 125400 | Mean(100): 144.3 | Eval: 178.2
Model saved to checkpoints/rainbow_space_invaders_ep100.pth

Episode 200 | Steps: 1234 | Score: 210.0 | Total Steps: 251200 | Mean(100): 178.2 | Eval: 224.1
New best model saved! Eval: 224.1
Model saved to checkpoints/rainbow_space_invaders_ep200.pth
```

### Training Complete

```
Episode 1000 | Steps: 1345 | Score: 850.0 | Total Steps: 1234500 | Mean(100): 756.3 | Eval: 892.5
New best model saved! Eval: 892.5
Model saved to checkpoints/rainbow_space_invaders_ep1000.pth
Model saved to checkpoints/rainbow_space_invaders_final.pth

============================================================
Training completed!
Final model saved to: checkpoints/rainbow_space_invaders_final.pth
Best model saved to: checkpoints/rainbow_space_invaders_best.pth
Logs saved to: logs/rainbow_space_invaders_20251210_150000.csv
============================================================
```

## Understanding the Output

### Single Line Format:
```
Episode 50 | Steps: 1247 | Score: 145.0 | Total Steps: 62350 | Mean(100): 132.4 | Eval: 156.3
```

Breaking it down:
- **Episode 50**: This is episode #50
- **Steps: 1247**: Episode lasted 1247 steps
- **Score: 145.0**: This episode got 145 points
- **Total Steps: 62350**: 62,350 total environment interactions so far
- **Mean(100): 132.4**: Average score over last 100 episodes is 132.4
- **Eval: 156.3**: Evaluation (5 test episodes) averaged 156.3 points

## Why This Format?

✅ **Clean**: No clutter, just essential info
✅ **Informative**: All key metrics in one line
✅ **Progress tracking**: Easy to see improvement over time
✅ **Complete data**: Full details still in CSV for analysis

## Monitoring Training

While training runs (silently between evaluations), you can:

### Check the CSV file:
```bash
tail -f logs/rainbow_space_invaders_*.csv
```

### Use TensorBoard:
```bash
tensorboard --logdir logs/tensorboard
```

### Count episodes completed:
```bash
wc -l logs/rainbow_space_invaders_*.csv
```

## Example Full Training Session (10 Episodes)

```
============================================================
Rainbow DQN Training on Space Invaders
============================================================

Creating environment...
State shape: (84, 84, 4)
Number of actions: 6
Device: cpu

Creating Rainbow DQN agent...
Creating prioritized replay buffer...
Setting up logging...

============================================================
Starting training...
============================================================

[Episodes 1-9 run silently...]

Episode 10 | Steps: 1147 | Score: 125.0 | Total Steps: 11470 | Mean(100): 125.0 | Eval: 118.3

Model saved to checkpoints/rainbow_space_invaders_ep10.pth
Model saved to checkpoints/rainbow_space_invaders_final.pth

============================================================
Training completed!
Final model saved to: checkpoints/rainbow_space_invaders_final.pth
Best model saved to: checkpoints/rainbow_space_invaders_best.pth
Logs saved to: logs/rainbow_space_invaders_20251210_150000.csv
============================================================
```

That's it! Clean, simple, informative. 🎯
