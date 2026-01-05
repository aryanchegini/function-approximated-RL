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
    'final_rainbow': '#3B82F6',
    'success': '#22C55E'
}

# File paths
BASE_DIR = Path("All results")
PLOT_DIR = BASE_DIR / "Plots"
PLOT_DIR.mkdir(exist_ok=True)

# Data files
FINAL_RAINBOW = BASE_DIR / "rainbow_files" / "rainbow_space_invaders_20251221_205859.csv"  # Rainbow DQN model
DQN_FILE = BASE_DIR / "DQN" / "dqn_space_invaders_20251222_142017.csv"
RANDOM_FILE = BASE_DIR / "Random" / "random_log.csv"

# PBT Rainbow files 
PBT_RAINBOW_DIR = BASE_DIR / "PBT Rainbow"
PBT_RAINBOW_FILES = [
    PBT_RAINBOW_DIR / f"member_{i}" / "episode_log.csv" 
    for i in range(4) 
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
    
    # Load Rainbow DQN
    data['final_rainbow'] = pd.read_csv(FINAL_RAINBOW)
    final_logged = len(data['final_rainbow'])
    final_actual = final_logged * 10
    print(f"Rainbow DQN: {final_actual:,} episodes, {data['final_rainbow']['total_steps'].iloc[-1]:,} steps")
    
    # Load PBT Rainbow members
    data['pbt_rainbow_members'] = []
    for i, file in enumerate(PBT_RAINBOW_FILES):
        if file.exists():
            df = pd.read_csv(file)
            data['pbt_rainbow_members'].append(df)
    print(f"PBT Rainbow: {len(data['pbt_rainbow_members'])} members loaded")
    
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


def analyze_rainbow_vs_pbt_rainbow(data):
    """Rainbow DQN vs Best PBT Member Comparison"""
    final_rainbow = data['final_rainbow']
    pbt_members = data['pbt_rainbow_members']
    
    if not pbt_members:
        print("Warning: No PBT Rainbow members found")
        return
    
    # Find best PBT member by final mean_return_100
    best_pbt_idx = max(range(len(pbt_members)), key=lambda i: pbt_members[i]['mean_return_100'].iloc[-1])
    best_pbt = pbt_members[best_pbt_idx]
    
    # Find common episode range for comparison
    min_episodes = min(len(final_rainbow), len(best_pbt))
    
    # Prepare data
    final_rainbow_plot = final_rainbow.iloc[:min_episodes].reset_index(drop=True)
    best_pbt_plot = best_pbt.iloc[:min_episodes].reset_index(drop=True)
    
    episodes = final_rainbow_plot['episode'].values
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # Plot 1: Mean(100) Comparison
    ax1 = axes[0]
    
    rainbow_smooth = rolling_mean_for_plot(final_rainbow_plot['mean_return_100'].values, window=50)
    best_pbt_smooth = rolling_mean_for_plot(best_pbt_plot['mean_return_100'].values, window=50)
    
    ax1.plot(episodes, rainbow_smooth, 
             label=f'Rainbow DQN (Mean(100)={final_rainbow_plot["mean_return_100"].iloc[-1]:.2f})', 
             color=COLORS['rainbow'], linewidth=3, alpha=0.9)
    ax1.plot(episodes, best_pbt_smooth, 
             label=f'Best PBT Member M{best_pbt_idx} (Mean(100)={best_pbt_plot["mean_return_100"].iloc[-1]:.2f})', 
             color=COLORS['success'], linewidth=3, alpha=0.9)
    
    ax1.set_xlabel('Episode', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Mean Return (100 episodes)', fontsize=13, fontweight='bold')
    ax1.set_title('Rainbow DQN vs Best PBT Member: Performance Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, loc='best')
    ax1.grid(True, alpha=0.3, linestyle=':')
   
    # Plot 2: Cumulative Returns
    ax2 = axes[1]
    
    rainbow_cumulative = np.cumsum(final_rainbow_plot['episode_return'].values)
    best_pbt_cumulative = np.cumsum(best_pbt_plot['episode_return'].values)
    
    ax2.plot(episodes, rainbow_cumulative, 
             label='Rainbow DQN', color=COLORS['rainbow'], linewidth=3, alpha=0.9)
    ax2.plot(episodes, best_pbt_cumulative, 
             label=f'Best PBT Member M{best_pbt_idx}', color=COLORS['success'], 
             linewidth=3, alpha=0.9)
    
    ax2.set_xlabel('Episode', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Cumulative Return', fontsize=13, fontweight='bold')
    ax2.set_title('Cumulative Performance Over Training', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11, loc='best')
    ax2.grid(True, alpha=0.3, linestyle=':')
    
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "1_rainbow_vs_best_pbt.png", bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"\n✓ Saved: 1_rainbow_vs_best_pbt.png")
    print(f"  Rainbow DQN final Mean(100): {final_rainbow_plot['mean_return_100'].iloc[-1]:.2f}")
    print(f"  Best PBT Member M{best_pbt_idx} final Mean(100): {best_pbt_plot['mean_return_100'].iloc[-1]:.2f}")
    
    return {
        'rainbow_final_mean100': final_rainbow_plot['mean_return_100'].iloc[-1],
        'best_pbt_final_mean100': best_pbt_plot['mean_return_100'].iloc[-1],
        'rainbow_mean': final_rainbow_plot['episode_return'].mean(),
        'best_pbt_mean': best_pbt_plot['episode_return'].mean(),
        'best_pbt_idx': best_pbt_idx
    }


def analyze_pbt_population(data):
    """Best PBT Member vs Population Average"""
    pbt_members = data['pbt_rainbow_members']
    
    if not pbt_members:
        print("Warning: No PBT Rainbow members found")
        return
    
    best_pbt_idx = max(range(len(pbt_members)), key=lambda i: pbt_members[i]['mean_return_100'].iloc[-1])
    best_pbt = pbt_members[best_pbt_idx]
    
    # Calculate population average
    min_episodes_pbt = min(len(df) for df in pbt_members)
    pbt_aligned = [df.iloc[:min_episodes_pbt].reset_index(drop=True) for df in pbt_members]
    
    # Average across all members
    pbt_avg_mean_100 = pd.DataFrame([df['mean_return_100'].values for df in pbt_aligned]).mean(axis=0)
    pbt_avg_episode_return = pd.DataFrame([df['episode_return'].values for df in pbt_aligned]).mean(axis=0)
    
    episodes = pbt_aligned[0]['episode'].values
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    # Plot 1: Mean(100) comparison
    ax1 = axes[0]
    
    best_smooth = rolling_mean_for_plot(best_pbt['mean_return_100'].iloc[:min_episodes_pbt].values, window=50)
    avg_smooth = rolling_mean_for_plot(pbt_avg_mean_100.values, window=50)
    
    ax1.plot(episodes, best_smooth,
            label=f'Best Member M{best_pbt_idx} (Mean(100)={best_pbt["mean_return_100"].iloc[:min_episodes_pbt].iloc[-1]:.2f})',
            color=COLORS['success'], linewidth=3, alpha=0.9)
    ax1.plot(episodes, avg_smooth,
            label=f'Population Average (Mean(100)={pbt_avg_mean_100.iloc[-1]:.2f})',
            color='#F59E0B', linewidth=3, alpha=0.9)
    
    ax1.set_xlabel('Episode', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Mean Return (100 episodes)', fontsize=13, fontweight='bold')
    ax1.set_title('PBT: Best Member vs Population Average', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, loc='best')
    ax1.grid(True, alpha=0.3, linestyle=':')
    
    # Plot 2: All 4 members' performance
    ax2 = axes[1]
    
    member_colors = ['#22C55E', '#3B82F6', '#F59E0B', '#EF4444']
    
    for idx, member in enumerate(pbt_members):
        member_smooth = rolling_mean_for_plot(member['mean_return_100'].iloc[:min_episodes_pbt].values, window=50)
        label_prefix = f'[Best] Member {idx}' if idx == best_pbt_idx else f'Member {idx}'
        ax2.plot(episodes, member_smooth,
                label=f'{label_prefix} (Mean(100)={member["mean_return_100"].iloc[:min_episodes_pbt].iloc[-1]:.2f})',
                color=member_colors[idx], linewidth=2 if idx == best_pbt_idx else 1.5, 
                alpha=0.9 if idx == best_pbt_idx else 0.8)
    
    ax2.set_xlabel('Episode', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Mean Return (100 episodes)', fontsize=13, fontweight='bold')
    ax2.set_title('PBT: All Members Performance', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11, loc='best')
    ax2.grid(True, alpha=0.3, linestyle=':')
    
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "2_pbt_best_vs_average.png", bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"\n✓ Saved: 2_pbt_best_vs_average.png")
    print(f"  Best Member M{best_pbt_idx} final Mean(100): {best_pbt['mean_return_100'].iloc[:min_episodes_pbt].iloc[-1]:.2f}")
    print(f"  Population Average final Mean(100): {pbt_avg_mean_100.iloc[-1]:.2f}")
    
    return {
        'best_pbt_idx': best_pbt_idx,
        'best_pbt_mean100': best_pbt['mean_return_100'].iloc[:min_episodes_pbt].iloc[-1],
        'population_avg_mean100': pbt_avg_mean_100.iloc[-1]
    }


def analyze_all_agents(data):
    """Rainbow vs PBT Rainbow vs DQN vs Random"""
    
    # Use rainbow for overall comparison
    rainbow = data['final_rainbow']
    dqn = data['dqn']
    random = data['random']
    pbt_members = data['pbt_rainbow_members']
    
    # Find best PBT member
    best_pbt = None
    best_pbt_idx = None
    if pbt_members:
        best_pbt_idx = max(range(len(pbt_members)), key=lambda i: pbt_members[i]['mean_return_100'].iloc[-1])
        best_pbt = pbt_members[best_pbt_idx]
    
    valid_agents = [df for df in [rainbow, best_pbt, dqn, random] if df is not None]
    if not valid_agents:
        return
    
    min_episodes = min(len(df) for df in valid_agents)
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    ax1 = axes[0]
    episodes = np.arange(min_episodes)
    
    window = 50
    
    if rainbow is not None:
        rainbow_smooth = rolling_mean_for_plot(rainbow['mean_return_100'].iloc[:min_episodes].values, window=window)
        ax1.plot(episodes, rainbow_smooth,
                label='Rainbow DQN', color=COLORS['rainbow'], linewidth=3, alpha=0.9)
    
    if best_pbt is not None:
        pbt_smooth = rolling_mean_for_plot(best_pbt['mean_return_100'].iloc[:min_episodes].values, window=window)
        ax1.plot(episodes, pbt_smooth,
                label=f'PBT Rainbow (M{best_pbt_idx})', color=COLORS['success'], linewidth=3, alpha=0.9)
    
    if dqn is not None:
        dqn_smooth = rolling_mean_for_plot(dqn['mean_return_100'].iloc[:min_episodes].values, window=window)
        ax1.plot(episodes, dqn_smooth,
                label='Standard DQN', color=COLORS['dqn'], linewidth=3, alpha=0.9)
    
    if random is not None:
        random_smooth = rolling_mean_for_plot(random['mean_return_100'].iloc[:min_episodes].values, window=window)
        ax1.plot(episodes, random_smooth,
                label='Random Agent', color=COLORS['random'], linewidth=3, alpha=0.9)
    
    ax1.set_xlabel('Episode', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Mean(100) Return', fontsize=13, fontweight='bold')
    ax1.set_title('All Agents: Performance Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Cumulative Episode Return
    ax2 = axes[1]
    
    if rainbow is not None:
        cumsum_rainbow = rainbow['episode_return'].iloc[:min_episodes].cumsum()
        ax2.plot(episodes, cumsum_rainbow.values,
                label='Rainbow DQN', color=COLORS['rainbow'], linewidth=3, alpha=0.9)
    
    if best_pbt is not None:
        cumsum_pbt = best_pbt['episode_return'].iloc[:min_episodes].cumsum()
        ax2.plot(episodes, cumsum_pbt.values,
                label=f'PBT Rainbow (M{best_pbt_idx})', color=COLORS['success'], linewidth=3, alpha=0.9)
    
    if dqn is not None:
        cumsum_dqn = dqn['episode_return'].iloc[:min_episodes].cumsum()
        ax2.plot(episodes, cumsum_dqn.values,
                label='Standard DQN', color=COLORS['dqn'], linewidth=3, alpha=0.9)
    
    if random is not None:
        cumsum_random = random['episode_return'].iloc[:min_episodes].cumsum()
        ax2.plot(episodes, cumsum_random.values,
                label='Random Agent', color=COLORS['random'], linewidth=3, alpha=0.9)
    
    ax2.set_xlabel('Episodes', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Cumulative Episode Return', fontsize=13, fontweight='bold')
    ax2.set_title('Cumulative Returns Over Training', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11, loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "3_all_agents_performance.png", bbox_inches='tight', dpi=300)
    plt.close()
    
    stats = {}
    if rainbow is not None:
        stats['rainbow'] = {
            'final_mean100': rainbow['mean_return_100'].iloc[-1],
            'mean_return': rainbow['episode_return'].mean(),
            'std_return': rainbow['episode_return'].std(),
            'max_return': rainbow['episode_return'].max()
        }
    if best_pbt is not None:
        stats['pbt_rainbow'] = {
            'final_mean100': best_pbt['mean_return_100'].iloc[-1],
            'mean_return': best_pbt['episode_return'].mean(),
            'std_return': best_pbt['episode_return'].std(),
            'max_return': best_pbt['episode_return'].max(),
            'member_idx': best_pbt_idx
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
    rainbow = data['final_rainbow']
    pbt_members = data['pbt_rainbow_members']
    
    # Find best PBT member (M3)
    best_pbt = None
    best_pbt_idx = None
    if pbt_members:
        best_pbt_idx = max(range(len(pbt_members)), key=lambda i: pbt_members[i]['mean_return_100'].iloc[-1])
        best_pbt = pbt_members[best_pbt_idx]
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    window = 100
    max_steps = None
    
    if rainbow is not None and 'avg_loss' in rainbow.columns:
        rainbow_loss = rainbow[rainbow['avg_loss'] > 0].copy()
        max_steps = rainbow_loss['total_steps'].iloc[-1]
        
        rainbow_loss['loss_smooth'] = rolling_mean_for_plot(rainbow_loss['avg_loss'].values, window=window)
        
        ax.plot(rainbow_loss['total_steps'], rainbow_loss['avg_loss'].values,
               color=COLORS['rainbow'], linewidth=0.5, alpha=0.2, label='Rainbow DQN Raw Loss')
        
        ax.plot(rainbow_loss['total_steps'], rainbow_loss['loss_smooth'],
               label=f'Rainbow DQN (Rolling Mean {window})', color=COLORS['rainbow'], 
               linewidth=2.5, alpha=0.9)
    
    if best_pbt is not None and 'avg_loss' in best_pbt.columns:
        pbt_loss = best_pbt[best_pbt['avg_loss'] > 0].copy()
        
        # Limit PBT to same max steps as Rainbow DQN
        if max_steps is not None:
            pbt_loss = pbt_loss[pbt_loss['total_steps'] <= max_steps].copy()
        
        pbt_loss['loss_smooth'] = rolling_mean_for_plot(pbt_loss['avg_loss'].values, window=window)
        
        ax.plot(pbt_loss['total_steps'], pbt_loss['avg_loss'].values,
               color=COLORS['success'], linewidth=0.5, alpha=0.2, label=f'PBT M{best_pbt_idx} Raw Loss')
        
        ax.plot(pbt_loss['total_steps'], pbt_loss['loss_smooth'],
               label=f'PBT Member M{best_pbt_idx} (Rolling Mean {window})', color=COLORS['success'], 
               linewidth=2.5, alpha=0.9)
    
    ax.set_xlabel('Total Steps', fontsize=13)
    ax.set_ylabel('Average Loss', fontsize=13)
    ax.set_title('Training Loss: Rainbow DQN vs PBT Rainbow', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "4_loss_analysis.png", bbox_inches='tight', dpi=300)
    plt.close()


def analyze_actions(data):
    """Action Distribution Analysis"""
    
    rainbow = data['final_rainbow']
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
    plt.savefig(PLOT_DIR / "5_action_distributions.png", bbox_inches='tight', dpi=300)
    plt.close()
    
    return action_counts


def analyze_pbt_hyperparameters(data):
    """PBT Hyperparameter Evolution"""
    pbt_members = data['pbt_rainbow_members']
    
    if not pbt_members:
        print("Warning: No PBT Rainbow members found for hyperparameter analysis")
        return
    
    # Load exploration logs for all PBT Rainbow members
    exploration_logs = []
    for i in range(len(pbt_members)):
        exp_file = PBT_RAINBOW_DIR / f"member_{i}" / "exploration_log.csv"
        if exp_file.exists():
            exploration_logs.append(pd.read_csv(exp_file))
        else:
            exploration_logs.append(None)
    
    # Key hyperparameters to track
    key_params = ['learning_rate', 'gamma', 'batch_size', 'alpha', 'sigma']
    
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)
    
    colors_pbt = ['#10B981', '#3B82F6', '#F59E0B', '#EF4444', '#8B5CF6']
    
    for param_idx, param_name in enumerate(key_params):
        if param_idx < 3:
            row = 0
            col = param_idx
        else:
            row = 1
            col = param_idx - 3
        ax = fig.add_subplot(gs[row, col])
        
        # Plot parameter value vs performance for each member
        for member_idx, (exp_log, member_data) in enumerate(zip(exploration_logs, pbt_members)):
            if exp_log is None:
                continue
            
            # Get parameter changes over time
            param_changes = exp_log[exp_log['param_name'] == param_name].copy()
            
            if len(param_changes) == 0:
                continue
            
            param_values = []
            eval_rewards = []
            
            for idx, row_data in param_changes.iterrows():
                episode = int(row_data['episode'])
                param_val = float(row_data['new_value'])
                
                # Find the closest episode in member_data to get episode return
                member_episodes = member_data['episode'].values
                closest_idx = np.argmin(np.abs(member_episodes - episode))
                eval_reward = member_data.iloc[closest_idx]['episode_return']
                
                param_values.append(param_val)
                eval_rewards.append(eval_reward)
            
            if len(param_values) > 0:
                ax.scatter(param_values, eval_rewards, 
                          s=40, alpha=0.7,
                          label=f'M{member_idx}', 
                          color=colors_pbt[member_idx])
        
        param_display = param_name.replace('_', ' ').title()
        ax.set_xlabel(param_display, fontsize=10, fontweight='bold')
        ax.set_ylabel('Episode Return', fontsize=10, fontweight='bold')
        ax.set_title(f'Performance vs {param_display}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3, linestyle=':')
    
    ax_events = fig.add_subplot(gs[1, 2])
    
    exploit_counts = []
    explore_counts = []
    for member_idx, exp_log in enumerate(exploration_logs):
        if exp_log is None:
            exploit_counts.append(0)
            explore_counts.append(0)
            continue
        exploit_counts.append(len(exp_log[exp_log['change_type'] == 'exploit']))
        explore_counts.append(len(exp_log[exp_log['change_type'] == 'explore']))
    
    num_members = len(exploration_logs)
    x_pos = np.arange(num_members)
    width = 0.35
    ax_events.bar(x_pos - width/2, exploit_counts, width, label='Exploit', color='#10B981', alpha=0.8)
    ax_events.bar(x_pos + width/2, explore_counts, width, label='Explore', color='#3B82F6', alpha=0.8)
    
    ax_events.set_xlabel('Member', fontsize=10)
    ax_events.set_ylabel('Event Count', fontsize=10)
    ax_events.set_title('PBT Actions per Member', fontsize=11, fontweight='bold')
    ax_events.set_xticks(x_pos)
    ax_events.set_xticklabels([f'M{i}' for i in range(num_members)], fontsize=9)
    ax_events.legend(fontsize=9)
    ax_events.grid(True, alpha=0.2, axis='y', linestyle=':')
    
    plt.suptitle('PBT Hyperparameter Analysis: Parameter Values vs Performance', fontsize=15, fontweight='bold', y=0.995)
    plt.savefig(PLOT_DIR / "6_pbt_hyperparameter_evolution.png", bbox_inches='tight', dpi=300)
    plt.close()


def analyze_pbt_hyperparameter_frequency(data):
    """Frequency distribution of hyperparameter values"""
    pbt_members = data['pbt_rainbow_members']
    
    if not pbt_members:
        print("Warning: No PBT Rainbow members found for hyperparameter frequency analysis")
        return
    
    exploration_logs = []
    for i in range(len(pbt_members)):
        exp_file = PBT_RAINBOW_DIR / f"member_{i}" / "exploration_log.csv"
        if exp_file.exists():
            exploration_logs.append(pd.read_csv(exp_file))
        else:
            exploration_logs.append(None)
    
    key_params = ['learning_rate', 'gamma', 'batch_size', 'alpha', 'sigma']
    
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)
    
    colors_pbt = ['#10B981', '#3B82F6', '#F59E0B', '#EF4444']
    
    for param_idx, param_name in enumerate(key_params):
        if param_idx < 3:
            row = 0
            col = param_idx
        else:
            row = 1
            col = param_idx - 3
        ax = fig.add_subplot(gs[row, col])
        
        for member_idx, exp_log in enumerate(exploration_logs):
            if exp_log is None:
                continue
            
            # Get parameter changes over time
            param_changes = exp_log[exp_log['param_name'] == param_name].copy()
            
            if len(param_changes) == 0:
                continue
            
            # Calculate duration for each parameter value
            param_values = []
            durations = []
            
            for i in range(len(param_changes)):
                current_episode = int(param_changes.iloc[i]['episode'])
                current_value = float(param_changes.iloc[i]['new_value'])
                
                # Calculate duration until next change (or end of training)
                if i < len(param_changes) - 1:
                    next_episode = int(param_changes.iloc[i + 1]['episode'])
                    duration = next_episode - current_episode
                else:
                    # Last value used until end of training (100k episodes)
                    duration = 100000 - current_episode
                
                if param_name == 'batch_size':
                    rounded_val = int(current_value)
                elif param_name in ['learning_rate', 'sigma']:
                    rounded_val = round(current_value, 6)
                else:
                    rounded_val = round(current_value, 4)
                
                param_values.append(rounded_val)
                durations.append(duration)
            
            if len(param_values) > 0:
                # Scatter plot: x=parameter value, y=duration (episodes)
                ax.scatter(param_values, durations, 
                          s=60,
                          alpha=0.7,
                          label=f'M{member_idx}', 
                          color=colors_pbt[member_idx])
        
        param_display = param_name.replace('_', ' ').title()
        ax.set_xlabel(f'{param_display} Value', fontsize=10, fontweight='bold')
        ax.set_ylabel('Duration (Episodes)', fontsize=10, fontweight='bold')
        ax.set_title(f'Usage Duration: {param_display}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3, linestyle=':')
        
        if param_name in ['learning_rate', 'gamma', 'sigma']:
            ax.tick_params(axis='x', rotation=45)
    
    plt.suptitle('PBT: Hyperparameter Value Usage Duration', fontsize=15, fontweight='bold', y=0.995)
    plt.savefig(PLOT_DIR / "6_pbt_hyperparameter_duration.png", bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"\n✓ Saved: 6_pbt_hyperparameter_duration.png")


def generate_summary_report(data, rainbow_pbt_stats, agent_stats, action_stats):
    report_path = PLOT_DIR / "comprehensive_analysis_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("Reinforcement Learning Experiments: Rainbow DQN, PBT Rainbow, DQN, and Random\n")
        f.write("="*80 + "\n\n")
        
        # Section 1: Rainbow vs PBT Rainbow
        f.write("1. RAINBOW DQN VS BEST PBT MEMBER\n")
        f.write("-"*80 + "\n")
        if rainbow_pbt_stats:
            f.write(f"Rainbow DQN Mean(100):              {rainbow_pbt_stats['rainbow_final_mean100']:.2f}\n")
            f.write(f"Best PBT Member Mean(100):          {rainbow_pbt_stats['best_pbt_final_mean100']:.2f}\n")
            f.write(f"\nRainbow DQN Average Return:         {rainbow_pbt_stats['rainbow_mean']:.2f}\n")
            f.write(f"Best PBT Average Return:            {rainbow_pbt_stats['best_pbt_mean']:.2f}\n")
            
            diff = rainbow_pbt_stats['rainbow_final_mean100'] - rainbow_pbt_stats['best_pbt_final_mean100']
            if diff > 0:
                f.write(f"\nRainbow DQN outperforms Best PBT by:    {diff:.2f} points\n")
            else:
                f.write(f"\nBest PBT outperforms Rainbow DQN by:    {abs(diff):.2f} points\n")
            
            f.write(f"Best PBT Member:                    Member {rainbow_pbt_stats['best_pbt_idx']}\n")
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
            if 'pbt_rainbow' in agent_stats and 'rainbow' in agent_stats:
                improvement = agent_stats['pbt_rainbow']['final_mean100'] / agent_stats['rainbow']['final_mean100']
                f.write(f"  PBT Rainbow vs Rainbow DQN:    {improvement:.2f}x better\n")
            if 'rainbow' in agent_stats and 'random' in agent_stats:
                improvement = agent_stats['rainbow']['final_mean100'] / agent_stats['random']['final_mean100']
                f.write(f"  Rainbow DQN vs Random:         {improvement:.2f}x better\n")
            if 'pbt_rainbow' in agent_stats and 'random' in agent_stats:
                improvement = agent_stats['pbt_rainbow']['final_mean100'] / agent_stats['random']['final_mean100']
                f.write(f"  PBT Rainbow vs Random:         {improvement:.2f}x better\n")
            if 'dqn' in agent_stats and 'random' in agent_stats:
                improvement = agent_stats['dqn']['final_mean100'] / agent_stats['random']['final_mean100']
                f.write(f"  DQN vs Random:                 {improvement:.2f}x better\n")
            if 'rainbow' in agent_stats and 'dqn' in agent_stats:
                improvement = agent_stats['rainbow']['final_mean100'] / agent_stats['dqn']['final_mean100']
                f.write(f"  Rainbow DQN vs DQN:            {improvement:.2f}x better\n")
            if 'pbt_rainbow' in agent_stats and 'dqn' in agent_stats:
                improvement = agent_stats['pbt_rainbow']['final_mean100'] / agent_stats['dqn']['final_mean100']
                f.write(f"  PBT Rainbow vs DQN:            {improvement:.2f}x better\n")
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
        'rainbow_vs_pbt_rainbow': rainbow_pbt_stats,
        'agent_statistics': agent_stats,
        'action_distributions': action_stats
    }
    json_path = PLOT_DIR / "analysis_data.json"
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)


def main():
    data = load_data()
    rainbow_pbt_stats = analyze_rainbow_vs_pbt_rainbow(data)
    pbt_population_stats = analyze_pbt_population(data)
    agent_stats = analyze_all_agents(data)
    analyze_loss(data)
    action_stats = analyze_actions(data)
    analyze_pbt_hyperparameter_frequency(data)  # Duration analysis only
    
    generate_summary_report(data, rainbow_pbt_stats, agent_stats, action_stats)
    
    print(f"\n{'='*80}")
    print(f"Complete. Saved to: {PLOT_DIR}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
