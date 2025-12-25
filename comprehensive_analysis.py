import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import ast
import re
from collections import Counter

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150  
plt.rcParams['savefig.dpi'] = 300  
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.figsize'] = (12, 8) 
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9

COLORS = {
    'rainbow': '#8B5CF6', 
    'pbt': '#10B981',      
    'dqn': '#F59E0B',     
    'random': '#EF4444',  
    'final_rainbow': '#3B82F6'
}

# File paths
BASE_DIR = Path("All results")
PLOT_DIR = BASE_DIR / "Plots"
PLOT_DIR.mkdir(exist_ok=True)

# Data files
BASELINE_RAINBOW = BASE_DIR / "rainbow_files" / "rainbow_space_invaders_20251221_205859.csv"
FINAL_RAINBOW = BASE_DIR / "rainbow_files" / "rainbow_space_invaders_20251221_205859.csv"  # Using same as baseline for now
DQN_FILE = BASE_DIR / "DQN" / "dqn_space_invaders_20251222_142017.csv"
RANDOM_FILE = BASE_DIR / "Random" / "random_log.csv"

# PBT files
PBT_DIR = BASE_DIR / "PBT"
PBT_FILES = [
    PBT_DIR / f"member_{i}" / "episode_log.csv" 
    for i in range(5)
]


def rolling_mean_for_plot(data, window=50):
    if len(data) < window:
        return data
    return pd.Series(data).rolling(window=window, min_periods=1, center=True).mean().values


def parse_actions(action_str):
    if pd.isna(action_str):
        return []
    
    action_str = str(action_str).strip()
    
    # Check if it contains tensor objects (DQN/Random format)
    if 'tensor' in action_str:
        # Extract numbers from tensor([[X]]) format
        numbers = re.findall(r'tensor\(\[\[(\d+)\]\]', action_str)
        return [int(n) for n in numbers]
    
    # Otherwise it's a simple list format (Rainbow)
    try:
        actions = ast.literal_eval(action_str)
        if isinstance(actions, list):
            return [int(a) for a in actions]
        return []
    except:
        return []


def load_data():
    """Load all experimental data"""
    data = {}
    
    data['baseline_rainbow'] = pd.read_csv(BASELINE_RAINBOW)
    rainbow_logged = len(data['baseline_rainbow'])
    rainbow_actual = rainbow_logged * 10
    print(f"Rainbow: {rainbow_actual:,} episodes, {data['baseline_rainbow']['total_steps'].iloc[-1]:,} steps")
    
    data['pbt_members'] = []
    for i, file in enumerate(PBT_FILES):
        if file.exists():
            df = pd.read_csv(file)
            data['pbt_members'].append(df)
    print(f"PBT: {len(data['pbt_members'])} members loaded")
    
    if DQN_FILE.exists():
        data['dqn'] = pd.read_csv(DQN_FILE)
        dqn_logged = len(data['dqn'])
        dqn_actual = dqn_logged * 10
        print(f"DQN: {dqn_actual:,} episodes")
    else:
        data['dqn'] = None
    
    if RANDOM_FILE.exists():
        data['random'] = pd.read_csv(RANDOM_FILE, header=None,
                                    names=['episode', 'total_steps', 'episode_return', 'episode_length',
                                           'mean_return_10', 'mean_return_100', 'col7', 'col8', 'actions_taken'])
        random_logged = len(data['random'])
        random_actual = random_logged * 10
        print(f"Random: {random_actual:,} episodes, {data['random']['total_steps'].iloc[-1]:,} steps")
    else:
        data['random'] = None
    
    print("\n" + "="*80)
    return data


def analyze_pbt_vs_baseline(data):
    """PBT vs Baseline Rainbow"""
    baseline = data['baseline_rainbow']
    pbt_members = data['pbt_members']
    
    if not pbt_members:
        return
    
    # Find minimum steps across all PBT members
    min_steps = min(df['total_steps'].iloc[-1] for df in pbt_members)
    
    baseline_truncated = baseline[baseline['total_steps'] <= min_steps].copy()
    pbt_truncated = [df[df['total_steps'] <= min_steps].copy() for df in pbt_members]
    
    min_episodes = min(
        min(len(df) for df in pbt_truncated),
        len(baseline_truncated)
    )
    
    pbt_aligned = [df.iloc[:min_episodes].reset_index(drop=True) for df in pbt_truncated]
    baseline_plot = baseline_truncated.iloc[:min_episodes].reset_index(drop=True)
    
    pbt_mean_returns = pd.DataFrame([df['mean_return_100'].values for df in pbt_aligned]).mean(axis=0)
    pbt_episode_returns = pd.DataFrame([df['episode_return'].values for df in pbt_aligned]).mean(axis=0)
    episodes = np.arange(len(pbt_mean_returns))
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax1 = axes[0]
    baseline_smooth = rolling_mean_for_plot(baseline_plot['mean_return_100'].values, window=10)
    pbt_smooth = rolling_mean_for_plot(pbt_mean_returns.values, window=10)
    
    ax1.plot(episodes, baseline_smooth, 
             label='Baseline Rainbow', color=COLORS['rainbow'], linewidth=2.5, alpha=0.9)
    ax1.plot(episodes, pbt_smooth, 
             label='PBT Rainbow (population mean)', color=COLORS['pbt'], linewidth=2.5, alpha=0.9)
    ax1.set_xlabel('Episode', fontsize=13)
    ax1.set_ylabel('Mean Return (100 episodes)', fontsize=13)
    ax1.set_title('PBT vs Baseline Rainbow: Mean(100) Performance', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
   
    ax2 = axes[1]
    baseline_ep_smooth = rolling_mean_for_plot(baseline_plot['episode_return'].values, window=20)
    pbt_ep_smooth = rolling_mean_for_plot(pbt_episode_returns.values, window=20)
    
    ax2.plot(episodes, baseline_ep_smooth, 
             label='Baseline Rainbow', color=COLORS['rainbow'], linewidth=2.5, alpha=0.9)
    ax2.plot(episodes, pbt_ep_smooth, 
             label='PBT Rainbow (population mean)', color=COLORS['pbt'], linewidth=2.5, alpha=0.9)
    ax2.set_xlabel('Episode', fontsize=13)
    ax2.set_ylabel('Episode Return', fontsize=13)
    ax2.set_title('PBT vs Baseline Rainbow: Episode Returns', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "1_pbt_vs_baseline.png", bbox_inches='tight', dpi=300)
    plt.close()
    
    return {
        'baseline_final_mean100': baseline_plot['mean_return_100'].iloc[-1],
        'pbt_final_mean100': pbt_mean_returns.iloc[-1],
        'baseline_mean': baseline_plot['episode_return'].mean(),
        'pbt_mean': pbt_episode_returns.mean()
    }


def analyze_all_agents(data):
    """Rainbow vs DQN vs Random"""
    
    # Use baseline rainbow as "final rainbow"
    rainbow = data['baseline_rainbow']
    dqn = data['dqn']
    random = data['random']
    
    valid_agents = [df for df in [rainbow, dqn, random] if df is not None]
    if not valid_agents:
        return
    
    min_episodes = min(len(df) for df in valid_agents)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    ax1 = axes[0]
    episodes = np.arange(min_episodes)
    
    window = 500
    
    if rainbow is not None:
        rainbow_smooth = rolling_mean_for_plot(rainbow['mean_return_100'].iloc[:min_episodes].values, window=window)
        ax1.plot(episodes, rainbow_smooth,
                label='Rainbow DQN', color=COLORS['rainbow'], linewidth=3, alpha=0.9)
    
    if dqn is not None:
        dqn_smooth = rolling_mean_for_plot(dqn['mean_return_100'].iloc[:min_episodes].values, window=window)
        ax1.plot(episodes, dqn_smooth,
                label='Standard DQN', color=COLORS['dqn'], linewidth=3, alpha=0.9)
    
    if random is not None:
        random_smooth = rolling_mean_for_plot(random['mean_return_100'].iloc[:min_episodes].values, window=window)
        ax1.plot(episodes, random_smooth,
                label='Random Agent', color=COLORS['random'], linewidth=2.5, linestyle='--', alpha=0.8)
    
    ax1.set_xlabel('Logged Episode (every 10 episodes)', fontsize=13)
    ax1.set_ylabel('Mean(100) Return', fontsize=13)
    ax1.set_title('Performance Comparison (Rolling Mean 500)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Cumulative Episode Return
    ax2 = axes[1]
    
    if rainbow is not None:
        cumsum_rainbow = rainbow['episode_return'].iloc[:min_episodes].cumsum()
        ax2.plot(episodes, cumsum_rainbow.values,
                label='Rainbow DQN', color=COLORS['rainbow'], linewidth=3, alpha=0.9)
    
    if dqn is not None:
        cumsum_dqn = dqn['episode_return'].iloc[:min_episodes].cumsum()
        ax2.plot(episodes, cumsum_dqn.values,
                label='Standard DQN', color=COLORS['dqn'], linewidth=3, alpha=0.9)
    
    if random is not None:
        cumsum_random = random['episode_return'].iloc[:min_episodes].cumsum()
        ax2.plot(episodes, cumsum_random.values,
                label='Random Agent', color=COLORS['random'], linewidth=2.5, linestyle='--', alpha=0.8)
    
    ax2.set_xlabel('Logged Episode (every 10 episodes)', fontsize=13)
    ax2.set_ylabel('Cumulative Episode Return', fontsize=13)
    ax2.set_title('Cumulative Returns Over Training', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "2_all_agents_performance.png", bbox_inches='tight', dpi=300)
    plt.close()
    
    stats = {}
    if rainbow is not None:
        stats['rainbow'] = {
            'final_mean100': rainbow['mean_return_100'].iloc[-1],
            'mean_return': rainbow['episode_return'].mean(),
            'std_return': rainbow['episode_return'].std(),
            'max_return': rainbow['episode_return'].max()
        }
    if dqn is not None:
        stats['dqn'] = {
            'final_mean100': dqn['mean_return_100'].iloc[-1],
            'mean_return': dqn['episode_return'].mean(),
            'std_return': dqn['episode_return'].std(),
            'max_return': dqn['episode_return'].max()
        }
    if random is not None:
        stats['random'] = {
            'final_mean100': random['mean_return_100'].iloc[-1],
            'mean_return': random['episode_return'].mean(),
            'std_return': random['episode_return'].std(),
            'max_return': random['episode_return'].max()
        }
    
    return stats


def analyze_loss(data):
    """Loss Analysis"""
    rainbow = data['baseline_rainbow']
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    if rainbow is not None and 'avg_loss' in rainbow.columns:
        rainbow_loss = rainbow[rainbow['avg_loss'] > 0].copy()
        
        window = 100
        rainbow_loss['loss_smooth'] = rolling_mean_for_plot(rainbow_loss['avg_loss'].values, window=window)
        
        ax.plot(rainbow_loss['total_steps'], rainbow_loss['avg_loss'].values,
               color=COLORS['rainbow'], linewidth=0.5, alpha=0.2, label='Raw Loss')
        
        ax.plot(rainbow_loss['total_steps'], rainbow_loss['loss_smooth'],
               label=f'Rainbow DQN (Rolling Mean {window})', color=COLORS['rainbow'], 
               linewidth=2.5, alpha=0.9)
    
    ax.set_xlabel('Total Steps', fontsize=13)
    ax.set_ylabel('Average Loss', fontsize=13)
    ax.set_title('Rainbow DQN Training Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "3_loss_analysis.png", bbox_inches='tight', dpi=300)
    plt.close()


def analyze_actions(data):
    """Action Distribution Analysis"""
    
    rainbow = data['baseline_rainbow']
    dqn = data['dqn']
    random = data['random']
    
    action_counts = {}
    
    for name, df in [('Rainbow', rainbow), ('DQN', dqn), ('Random', random)]:
        if df is None or 'actions_taken' not in df.columns:
            continue
        
        all_actions = []
        for actions_str in df['actions_taken']:
            actions = parse_actions(actions_str)
            all_actions.extend(actions)
        
        if all_actions:
            counts = Counter(all_actions)
            total = sum(counts.values())
            action_counts[name] = {k: v/total for k, v in sorted(counts.items())}
    
    if not action_counts:
        print("   ERROR: No action data could be parsed!")
        return
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    x = np.arange(6)  # Space Invaders has 6 actions
    width = 0.25
    
    if 'Rainbow' in action_counts:
        rainbow_vals = [action_counts['Rainbow'].get(i, 0) for i in range(6)]
        ax.bar(x - width, rainbow_vals, width, label='Rainbow DQN', 
               color=COLORS['rainbow'], alpha=0.8, edgecolor='black', linewidth=1.2)
    
    if 'DQN' in action_counts:
        dqn_vals = [action_counts['DQN'].get(i, 0) for i in range(6)]
        ax.bar(x, dqn_vals, width, label='Standard DQN', 
               color=COLORS['dqn'], alpha=0.8, edgecolor='black', linewidth=1.2)
    
    if 'Random' in action_counts:
        random_vals = [action_counts['Random'].get(i, 0) for i in range(6)]
        ax.bar(x + width, random_vals, width, label='Random Agent', 
               color=COLORS['random'], alpha=0.8, edgecolor='black', linewidth=1.2)
    
    ax.set_xlabel('Action', fontsize=13)
    ax.set_ylabel('Normalized Frequency', fontsize=13)
    ax.set_title('Action Distribution Comparison (Normalized)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE'], rotation=45, ha='right')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "4_action_distributions.png", bbox_inches='tight', dpi=300)
    plt.close()
    
    return action_counts


def analyze_pbt_hyperparameters(data):
    """PBT Hyperparameter Evolution"""
    pbt_members = data['pbt_members']
    
    if not pbt_members:
        return
    
    # Load exploration logs for all members
    exploration_logs = []
    for i in range(5):
        exp_file = PBT_DIR / f"member_{i}" / "exploration_log.csv"
        if exp_file.exists():
            exploration_logs.append(pd.read_csv(exp_file))
        else:
            exploration_logs.append(None)
    
    # Key hyperparameters to track
    key_params = ['learning_rate', 'gamma', 'batch_size', 'alpha', 'sigma']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    colors_pbt = ['#10B981', '#3B82F6', '#F59E0B', '#EF4444', '#8B5CF6']
    
    for param_idx, param_name in enumerate(key_params):
        ax = axes[param_idx]
        
        # Plot evolution for each member
        for member_idx, exp_log in enumerate(exploration_logs):
            if exp_log is None:
                continue
            
            # Get parameter changes over time
            param_changes = exp_log[exp_log['param_name'] == param_name].copy()
            
            if len(param_changes) == 0:
                continue
            
            # Build timeline of parameter values
            episodes = [0]
            values = [float(param_changes.iloc[0]['new_value'])]
            
            for idx, row in param_changes.iloc[1:].iterrows():
                episodes.append(int(row['episode']))
                values.append(float(row['new_value']))
            
            ax.plot(episodes, values, label=f'Member {member_idx}', 
                   color=colors_pbt[member_idx], linewidth=2, alpha=0.8, marker='o', markersize=4)
        
        param_display = param_name.replace('_', ' ').title()
        ax.set_xlabel('Episode', fontsize=11)
        ax.set_ylabel(param_display, fontsize=11)
        ax.set_title(f'{param_display} Evolution', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
    
    fig.delaxes(axes[5])
    
    plt.suptitle('PBT Hyperparameter Evolution Across Population', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "5_pbt_hyperparameter_evolution.png", bbox_inches='tight', dpi=300)
    plt.close()


def generate_summary_report(data, pbt_stats, agent_stats, action_stats):
    report_path = PLOT_DIR / "comprehensive_analysis_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("Reinforcement Learning Experiments: Rainbow DQN, PBT, DQN, and Random Agents\n")
        f.write("="*80 + "\n\n")
        
        # Section 1: PBT vs Baseline
        f.write("1. PBT VS BASELINE RAINBOW\n")
        f.write("-"*80 + "\n")
        if pbt_stats:
            f.write(f"Baseline Rainbow Final Mean(100):  {pbt_stats['baseline_final_mean100']:.2f}\n")
            f.write(f"PBT Population Final Mean(100):    {pbt_stats['pbt_final_mean100']:.2f}\n")
            f.write(f"Baseline Average Return:            {pbt_stats['baseline_mean']:.2f}\n")
            f.write(f"PBT Average Return:                 {pbt_stats['pbt_mean']:.2f}\n")
            improvement = (pbt_stats['baseline_final_mean100'] / pbt_stats['pbt_final_mean100'] - 1) * 100
            f.write(f"\nBaseline outperforms PBT by:        {improvement:.1f}%\n")
        f.write("\n")
        
        # Section 2: All Agents Comparison
        f.write("2. ALL AGENTS PERFORMANCE COMPARISON\n")
        f.write("-"*80 + "\n")
        if agent_stats:
            for agent_name, stats in agent_stats.items():
                f.write(f"\n{agent_name.upper()}:\n")
                f.write(f"  Final Mean(100):     {stats['final_mean100']:.2f}\n")
                f.write(f"  Mean Return:         {stats['mean_return']:.2f} ± {stats['std_return']:.2f}\n")
                f.write(f"  Max Return:          {stats['max_return']:.2f}\n")
            
            # Improvements
            f.write("\nRELATIVE IMPROVEMENTS:\n")
            if 'rainbow' in agent_stats and 'random' in agent_stats:
                improvement = agent_stats['rainbow']['final_mean100'] / agent_stats['random']['final_mean100']
                f.write(f"  Rainbow vs Random:   {improvement:.2f}x better\n")
            if 'dqn' in agent_stats and 'random' in agent_stats:
                improvement = agent_stats['dqn']['final_mean100'] / agent_stats['random']['final_mean100']
                f.write(f"  DQN vs Random:       {improvement:.2f}x better\n")
            if 'rainbow' in agent_stats and 'dqn' in agent_stats:
                improvement = agent_stats['rainbow']['final_mean100'] / agent_stats['dqn']['final_mean100']
                f.write(f"  Rainbow vs DQN:      {improvement:.2f}x better\n")
        f.write("\n")
        
        # Section 3: Action Distributions
        f.write("3. ACTION DISTRIBUTION ANALYSIS\n")
        f.write("-"*80 + "\n")
        if action_stats:
            action_names = ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']
            for agent_name, actions in action_stats.items():
                f.write(f"\n{agent_name}:\n")
                for action_id in range(6):
                    freq = actions.get(action_id, 0)
                    f.write(f"  {action_names[action_id]:12s}: {freq*100:5.2f}%\n")
        f.write("\n")
    
    json_data = {
        'pbt_comparison': pbt_stats,
        'agent_statistics': agent_stats,
        'action_distributions': action_stats
    }
    json_path = PLOT_DIR / "analysis_data.json"
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)


def main():
    data = load_data()
    pbt_stats = analyze_pbt_vs_baseline(data)
    agent_stats = analyze_all_agents(data)
    analyze_loss(data)
    action_stats = analyze_actions(data)
    analyze_pbt_hyperparameters(data)
    
    generate_summary_report(data, pbt_stats, agent_stats, action_stats)
    
    print(f"\nComplete. Saved to: {PLOT_DIR}")


if __name__ == "__main__":
    main()
