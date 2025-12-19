
``pbt_checkpoints/
├── best_model.pt                    # Global best model
├── best_metadata.json               # Metadata for global best
├── checkpoints_log.csv              # Log of all periodic checkpoints
├── checkpoints/
│   ├── checkpoint_01.pt
│   ├── checkpoint_02.pt
│   └── ... (up to checkpoint_10.pt)
└── member_0/
    ├── episode_log.csv              # Episode metrics
    ├── exploration_log.csv          # Hyperparameter changes
    └── configs/
        ├── config_init.json         # Initial config
        ├── config_exploit_ep80_step19200.json
        └── config_explore_ep120_step28800.json
```

## Usage

### 1. Member Initialization

The `PBTLogger` is automatically created when you instantiate a `Member`:

```python
member = Member(id=0)
# Logger is initialized automatically
# Initial config is logged to exploration_log.csv
```

### 2. Logging Episodes

Log episode metrics every N episodes:

```python
if episode % LOGGING_CONFIG['episode_log_frequency'] == 0:
    member.logger.log_episode(
        episode=episode,
        total_steps=total_steps,
        episode_return=episode_reward,
        episode_length=episode_steps,
        mean_return_10=mean_return_10,
        mean_return_100=mean_return_100,
        avg_loss=avg_loss,
        buffer_size=len(member.replay_buffer)
    )
```

### 3. Logging Exploitation

When a member copies another member's config:

```python
member.exploit(better_member, episode=episode, total_steps=total_steps)
# Automatically logs:
# - Which member was copied
# - What parameters changed
# - Config snapshot saved to configs/
```

### 4. Logging Exploration

When hyperparameters are randomly perturbed:

```python
member.explore(episode=episode, total_steps=total_steps)
# Automatically logs:
# - What parameters changed
# - Old and new values
# - Config snapshot saved to configs/
```

### 5. Global Best Model

Automatically saves when a new best is found:

```python
if checkpoint_manager.update_best(
    member_id=member.id,
    episode=episode,
    total_steps=total_steps,
    score=eval_score,
    agent=member.agent,
    config=member.config
):
    print("New global best saved!")
```

### 6. Periodic Checkpoints

Saves snapshots at regular intervals (e.g., every 10% of training):

```python
if checkpoint_manager.should_save_checkpoint(total_steps):
    checkpoint_manager.save_checkpoint(
        member_id=best_member_id,
        episode=episode,
        total_steps=total_steps,
        score=score,
        agent=best_agent,
        config=config
    )
```

## Configuration

Add to `SpaceInvadersConfig.py`:

```python
LOGGING_CONFIG = {
    'checkpoint_dir': 'pbt_checkpoints',
    'episode_log_frequency': 10,        # Log every N episodes
    'save_global_best': True,
    'save_periodic_checkpoints': True,
    'num_checkpoints': 10,              # Number of progress snapshots
    'checkpoint_by': 'steps',           # 'steps' or 'episodes'
}

TRAINING_CONFIG = {
    'total_training_steps': 10_000_000,  # Total steps for checkpoint intervals
    ...
}
```

## CSV Formats

### episode_log.csv
```csv
timestamp,episode,total_steps,episode_return,episode_length,mean_return_10,mean_return_100,avg_loss,buffer_size
2025-12-19T14:20:00,100,24300,145.5,243,132.3,128.7,0.0234,24300
```

### exploration_log.csv
```csv
timestamp,change_type,source_member,episode,total_steps,param_name,old_value,new_value
2025-12-19T14:20:30,exploit,3,80,19200,learning_rate,0.000123,0.000089
2025-12-19T14:20:30,explore,,80,19200,gamma,0.99,0.995
```

### checkpoints_log.csv
```csv
checkpoint_num,timestamp,member_id,episode,total_steps,eval_score,model_path
1,2025-12-19T14:30:00,3,250,100000,345.2,checkpoints/checkpoint_01.pt
```

## Key Features

✅ **Simple API** - Just call logger methods, everything else is automatic
✅ **Clean structure** - Per-member directories keep logs organized
✅ **Full provenance** - Track every config change with snapshots
✅ **Progress tracking** - Periodic checkpoints show training evolution
✅ **Global best** - Always save the best model across all agents
✅ **Multiprocess-safe** - Each member writes to its own files

## Tips

1. **Episode log frequency**: Start with 10-20 episodes for development, increase for long runs
2. **Config snapshots**: Full config saved as JSON for easy inspection
3. **Periodic checkpoints**: Shows how best agent improves over time
4. **Analysis**: Use pandas to easily load and analyze CSV logs

```python
import pandas as pd

# Load episode metrics
df = pd.read_csv('pbt_checkpoints/member_0/episode_log.csv')
print(df.describe())

# Load exploration events
exp_df = pd.read_csv('pbt_checkpoints/member_0/exploration_log.csv')
print(exp_df[exp_df['change_type'] == 'exploit'])
```

## Example Output

When running, you'll see:
```
Agent 0 | Episode 100/1000 | Reward: 145.50 | Mean(100): 128.7
Agent 0 Eval after 80 episodes: Average Reward: 156.3
Agent 0 Exploiting member 3 with score 178.90
Agent 0 New config: {'learning_rate': 8.9e-05, 'gamma': 0.995, ...}
New global best model saved! Score: 178.90
Saved periodic checkpoint at 1000000 steps (best: member 3)
```

And logs in:
- `pbt_checkpoints/member_0/episode_log.csv`
- `pbt_checkpoints/member_0/exploration_log.csv`
- `pbt_checkpoints/best_model.pt`
- `pbt_checkpoints/checkpoints/checkpoint_01.pt`
