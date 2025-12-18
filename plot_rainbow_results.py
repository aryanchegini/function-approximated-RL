import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ast import literal_eval
import warnings
import os
from pathlib import Path
warnings.filterwarnings('ignore')

# Configuration
CSV_PATH = "vast_models/rainbow_space_invaders_20251217_144501.csv"

# Create results directory structure based on CSV filename
csv_filename = Path(CSV_PATH).stemt
RESULTS_DIR = os.path.join("results", csv_filename)
os.makedirs(RESULTS_DIR, exist_ok=True)
print(f"Results will be saved to: {os.path.abspath(RESULTS_DIR)}/")

# Load and preprocess data
df = pd.read_csv(CSV_PATH)

df['actions_parsed'] = df['actions_taken'].apply(literal_eval)
df['action_count'] = df['actions_parsed'].apply(len)
df['episode_index'] = range(len(df))

def calculate_entropy(actions):
    unique, counts = np.unique(actions, return_counts=True)
    probs = counts / counts.sum()
    entropy = -np.sum(probs * np.log2(probs + 1e-10))
    return entropy

df['action_entropy'] = df['actions_parsed'].apply(calculate_entropy)
df['action_diversity'] = df['actions_parsed'].apply(lambda x: len(set(x)))

plt.style.use('seaborn-v0_8-darkgrid')
fig_num = 1

# 1. Training Performance
plt.figure(fig_num, figsize=(12, 5))
fig_num += 1

plt.subplot(1, 2, 1)
plt.plot(df['total_steps'], df['episode_return'], alpha=0.2, label='Raw returns', color='blue')
plt.plot(df['total_steps'], df['episode_return'].rolling(window=500, min_periods=1).mean(), 
         linewidth=2, label='Rolling mean (500 eps)', color='red')
plt.xlabel('Total Environment Steps')
plt.ylabel('Episode Return')
plt.title('Episode Return vs Timesteps')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(df['total_steps'], df['episode_return'], alpha=0.2, label='Raw returns', color='blue')
plt.plot(df['total_steps'], df['episode_return'].rolling(window=200, min_periods=1).mean(), 
         linewidth=2, label='Rolling mean (200 eps)', color='red')
plt.xlabel('Total Environment Steps')
plt.ylabel('Episode Return')
plt.title('Episode Return (Shorter Window)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, '1_training_performance.png'), dpi=300, bbox_inches='tight')

# 2. Learning Stability
plt.figure(fig_num, figsize=(12, 5))
fig_num += 1

plt.subplot(1, 2, 1)
plt.plot(df['total_steps'], df['mean_return_100'], label='Mean return (100 eps)', linewidth=2, alpha=0.8)
plt.plot(df['total_steps'], df['episode_return'].rolling(window=500, min_periods=1).mean(), 
         label='Mean return (500 eps)', linewidth=2, alpha=0.8)
plt.xlabel('Total Environment Steps')
plt.ylabel('Mean Episode Return')
plt.title('Rolling Mean Returns')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(df['total_steps'], df['episode_length'], alpha=0.2, color='green')
plt.plot(df['total_steps'], df['episode_length'].rolling(window=200, min_periods=1).mean(), 
         linewidth=2, label='Rolling mean (200 eps)', color='darkgreen')
plt.xlabel('Total Environment Steps')
plt.ylabel('Episode Length')
plt.title('Episode Length')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, '2_learning_stability.png'), dpi=300, bbox_inches='tight')

# 3. Optimization Behavior
plt.figure(fig_num, figsize=(12, 5))
fig_num += 1

plt.subplot(1, 2, 1)
plt.plot(df['total_steps'], df['avg_loss'], alpha=0.2, color='orange')
plt.plot(df['total_steps'], df['avg_loss'].rolling(window=200, min_periods=1).mean(), 
         linewidth=2, label='Rolling mean (200 eps)', color='darkorange')
plt.xlabel('Total Environment Steps')
plt.ylabel('Average Loss')
plt.title('Training Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(df['total_steps'], df['buffer_size'], linewidth=2, color='purple')
plt.xlabel('Total Environment Steps')
plt.ylabel('Replay Buffer Size')
plt.title('Replay Buffer Growth')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, '3_optimization_behavior.png'), dpi=300, bbox_inches='tight')

# 4. Sample Efficiency
plt.figure(fig_num, figsize=(12, 5))
fig_num += 1

plt.subplot(1, 2, 1)
plt.plot(df['total_steps'], df['episode_return'], alpha=0.2, color='blue')
plt.plot(df['total_steps'], df['episode_return'].rolling(window=500, min_periods=1).mean(), 
         linewidth=2, label='Rolling mean (500 eps)', color='red')
plt.xlabel('Total Environment Steps')
plt.ylabel('Episode Return')
plt.title('Return vs Timesteps')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
steps_per_episode = df['total_steps'].diff().fillna(df['total_steps'].iloc[0])
plt.plot(df['total_steps'], steps_per_episode.rolling(window=200, min_periods=1).mean(), 
         linewidth=2, color='teal', label='Rolling mean (200 eps)')
plt.xlabel('Total Environment Steps')
plt.ylabel('Steps per Episode')
plt.title('Steps per Episode Over Time')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, '4_sample_efficiency.png'), dpi=300, bbox_inches='tight')

# 5. Action Space Analysis
all_actions = []
for actions in df['actions_parsed']:
    all_actions.extend(actions)

plt.figure(fig_num, figsize=(14, 10))
fig_num += 1

plt.subplot(2, 2, 1)
unique_actions, counts = np.unique(all_actions, return_counts=True)
plt.bar(unique_actions, counts, color='steelblue', edgecolor='black')
plt.xlabel('Action ID')
plt.ylabel('Frequency')
plt.title('Global Action Frequency Distribution')
plt.xticks(unique_actions)
plt.grid(True, alpha=0.3, axis='y')

plt.subplot(2, 2, 2)
window_size = 100
action_ids = sorted(set(all_actions))
cumulative_actions = {a: [] for a in action_ids}

for idx in range(len(df)):
    start_idx = max(0, idx - window_size + 1)
    window_actions = []
    for i in range(start_idx, idx + 1):
        window_actions.extend(df['actions_parsed'].iloc[i])
    
    action_counts = {a: window_actions.count(a) for a in action_ids}
    total = len(window_actions)
    
    for a in action_ids:
        cumulative_actions[a].append(action_counts[a] / total if total > 0 else 0)

bottom = np.zeros(len(df))
colors = plt.cm.tab10(np.linspace(0, 1, len(action_ids)))
for i, action_id in enumerate(action_ids):
    plt.fill_between(df['episode_index'], bottom, 
                     bottom + cumulative_actions[action_id], 
                     label=f'Action {action_id}', alpha=0.7, color=colors[i])
    bottom += cumulative_actions[action_id]

plt.xlabel('Episode Index')
plt.ylabel('Action Proportion')
plt.title(f'Action Distribution Over Time (Rolling {window_size} episodes)')
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=8)
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 3)
dominant_actions = df['actions_parsed'].apply(lambda x: max(set(x), key=x.count) if len(x) > 0 else -1)
plt.scatter(df['episode_index'], dominant_actions, alpha=0.5, s=10, color='crimson')
plt.xlabel('Episode Index')
plt.ylabel('Dominant Action ID')
plt.title('Most Frequent Action per Episode')
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 4)
plt.plot(df['episode_index'], df['action_diversity'], alpha=0.4, color='purple')
plt.plot(df['episode_index'], df['action_diversity'].rolling(window=100, min_periods=1).mean(), 
         linewidth=2, label='Rolling mean (100)', color='darkviolet')
plt.xlabel('Episode Index')
plt.ylabel('Unique Actions per Episode')
plt.title('Action Diversity Over Time')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, '5_action_space_analysis.png'), dpi=300, bbox_inches='tight')

print(f"\nAction Statistics:")
print(f"  Total actions taken: {len(all_actions):,}")
print(f"  Unique action IDs: {sorted(unique_actions.tolist())}")
print(f"  Action frequencies:")
for action_id, count in zip(unique_actions, counts):
    percentage = 100 * count / len(all_actions)
    print(f"    Action {action_id}: {count:,} ({percentage:.2f}%)")

dominant_action = unique_actions[np.argmax(counts)]
print(f"  Dominant action: {dominant_action} ({100*counts.max()/len(all_actions):.2f}%)")

# 6. Policy Behavior Diagnostics
plt.figure(fig_num, figsize=(12, 5))
fig_num += 1

plt.subplot(1, 2, 1)
plt.plot(df['episode_index'], df['action_entropy'], alpha=0.4, color='teal')
plt.plot(df['episode_index'], df['action_entropy'].rolling(window=100, min_periods=1).mean(), 
         linewidth=2, label='Rolling mean (100)', color='darkcyan')
plt.xlabel('Episode Index')
plt.ylabel('Action Entropy (bits)')
plt.title('Action Entropy Over Time')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(df['action_entropy'], df['episode_return'], alpha=0.3, s=20, color='coral')
z = np.polyfit(df['action_entropy'], df['episode_return'], 1)
p = np.poly1d(z)
plt.plot(df['action_entropy'], p(df['action_entropy']), "r--", linewidth=2, 
         label=f'Linear fit (r={np.corrcoef(df["action_entropy"], df["episode_return"])[0,1]:.3f})')
plt.xlabel('Action Entropy (bits)')
plt.ylabel('Episode Return')
plt.title('Return vs Action Diversity')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, '6_policy_diagnostics.png'), dpi=300, bbox_inches='tight')

correlation = np.corrcoef(df['action_entropy'], df['episode_return'])[0, 1]
print(f"\nCorrelation between action entropy and return: {correlation:.3f}")

# 7. Final Performance Summary
last_100_returns = df['episode_return'].iloc[-100:] if len(df) >= 100 else df['episode_return']
final_mean_return = last_100_returns.mean()
final_std_return = last_100_returns.std()
best_return = df['episode_return'].max()
best_episode = df.loc[df['episode_return'].idxmax(), 'episode_index']
worst_return = df['episode_return'].min()
median_return = df['episode_return'].median()

initial_mean = df['episode_return'].iloc[:100].mean() if len(df) >= 100 else df['episode_return'].iloc[:10].mean()
improvement = final_mean_return - initial_mean
improvement_pct = 100 * improvement / (abs(initial_mean) + 1e-10)

print(f"\nPerformance Metrics:")
print(f"  Total episodes: {len(df)}")
print(f"  Total environment steps: {df['total_steps'].iloc[-1]:,}")
print(f"  Final mean return (last 100 eps): {final_mean_return:.2f} ± {final_std_return:.2f}")
print(f"  Best episode return: {best_return:.2f} (episode {best_episode})")
print(f"  Worst episode return: {worst_return:.2f}")
print(f"  Median episode return: {median_return:.2f}")
print(f"  Overall mean return: {df['episode_return'].mean():.2f}")
print(f"  Improvement from start: {improvement:+.2f} ({improvement_pct:+.1f}%)")
print(f"  Final stability (std/mean): {final_std_return/abs(final_mean_return + 1e-10):.3f}")

print(f"\nEpisode Statistics:")
print(f"  Mean episode length: {df['episode_length'].mean():.1f} ± {df['episode_length'].std():.1f}")
print(f"  Median episode length: {df['episode_length'].median():.1f}")
print(f"  Min/Max episode length: {df['episode_length'].min()} / {df['episode_length'].max()}")

print(f"\nOptimization Statistics:")
print(f"  Final average loss: {df['avg_loss'].iloc[-100:].mean():.4f}")
print(f"  Final buffer size: {df['buffer_size'].iloc[-1]:,}")

plt.figure(fig_num, figsize=(14, 6))
fig_num += 1

plt.subplot(1, 3, 1)
plt.hist(df['episode_return'], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
plt.axvline(final_mean_return, color='red', linestyle='--', linewidth=2, label=f'Final mean: {final_mean_return:.2f}')
plt.axvline(best_return, color='green', linestyle='--', linewidth=2, label=f'Best: {best_return:.2f}')
plt.xlabel('Episode Return')
plt.ylabel('Frequency')
plt.title('Distribution of Episode Returns')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

plt.subplot(1, 3, 2)
plt.plot(df['episode_index'], df['mean_return_100'], linewidth=2, color='blue', label='Mean return (100)')
plt.fill_between(df['episode_index'], 
                 df['mean_return_100'] - df['episode_return'].rolling(100, min_periods=1).std(),
                 df['mean_return_100'] + df['episode_return'].rolling(100, min_periods=1).std(),
                 alpha=0.3, color='blue', label='±1 std')
plt.xlabel('Episode Index')
plt.ylabel('Mean Return')
plt.title('Learning Curve with Uncertainty')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
quartile_size = len(df) // 4
quartiles = []
labels = []
for i in range(4):
    start = i * quartile_size
    end = (i + 1) * quartile_size if i < 3 else len(df)
    quartiles.append(df['episode_return'].iloc[start:end])
    labels.append(f'Q{i+1}\n({start}-{end})')

plt.boxplot(quartiles, labels=labels)
plt.ylabel('Episode Return')
plt.title('Performance by Training Quarter')
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, '7_final_summary.png'), dpi=300, bbox_inches='tight')
