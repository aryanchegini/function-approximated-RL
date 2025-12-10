"""
Plot training progress from CSV logs.
"""
import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_training_progress(csv_path: str, save_path: str = None):
    """
    Plot training metrics from CSV log.
    
    Args:
        csv_path: Path to CSV log file
        save_path: Path to save plot (optional)
    """
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return
    
    # Load data
    df = pd.read_csv(csv_path)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Rainbow DQN Training Progress - Space Invaders', fontsize=16)
    
    # Plot 1: Episode Returns
    ax = axes[0, 0]
    ax.plot(df['episode'], df['episode_return'], alpha=0.3, label='Episode Return')
    ax.plot(df['episode'], df['mean_return_10'], alpha=0.8, label='Mean (10 episodes)')
    if 'mean_return_100' in df.columns:
        ax.plot(df['episode'], df['mean_return_100'], alpha=0.8, label='Mean (100 episodes)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Return')
    ax.set_title('Episode Returns')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Episode Length
    ax = axes[0, 1]
    ax.plot(df['episode'], df['episode_length'], alpha=0.5)
    # Add smoothed line
    window = min(20, len(df) // 10)
    if window > 0:
        smoothed = df['episode_length'].rolling(window=window).mean()
        ax.plot(df['episode'], smoothed, linewidth=2, label=f'Smoothed ({window} episodes)')
        ax.legend()
    ax.set_xlabel('Episode')
    ax.set_ylabel('Length (steps)')
    ax.set_title('Episode Length')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Training Loss
    ax = axes[1, 0]
    if 'avg_loss' in df.columns:
        # Remove zeros (episodes where no training happened)
        loss_data = df[df['avg_loss'] > 0]
        ax.plot(loss_data['episode'], loss_data['avg_loss'], alpha=0.5)
        # Add smoothed line
        window = min(20, len(loss_data) // 10)
        if window > 0:
            smoothed = loss_data['avg_loss'].rolling(window=window).mean()
            ax.plot(loss_data['episode'], smoothed, linewidth=2, label=f'Smoothed ({window} episodes)')
            ax.legend()
    ax.set_xlabel('Episode')
    ax.set_ylabel('Loss')
    ax.set_title('Average Training Loss')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Buffer Size and Total Steps
    ax = axes[1, 1]
    ax2 = ax.twinx()
    
    if 'buffer_size' in df.columns:
        line1 = ax.plot(df['episode'], df['buffer_size'], 'b-', alpha=0.7, label='Buffer Size')
    if 'total_steps' in df.columns:
        line2 = ax2.plot(df['episode'], df['total_steps'], 'r-', alpha=0.7, label='Total Steps')
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Buffer Size', color='b')
    ax2.set_ylabel('Total Steps', color='r')
    ax.set_title('Buffer Size and Total Steps')
    ax.tick_params(axis='y', labelcolor='b')
    ax2.tick_params(axis='y', labelcolor='r')
    ax.grid(True, alpha=0.3)
    
    # Combine legends
    lines = line1 + line2 if 'buffer_size' in df.columns and 'total_steps' in df.columns else []
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper left')
    
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def plot_comparison(csv_paths: list, labels: list, save_path: str = None):
    """
    Plot comparison of multiple training runs.
    
    Args:
        csv_paths: List of paths to CSV log files
        labels: List of labels for each run
        save_path: Path to save plot (optional)
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for csv_path, label in zip(csv_paths, labels):
        if not os.path.exists(csv_path):
            print(f"Warning: CSV file not found at {csv_path}")
            continue
        
        df = pd.read_csv(csv_path)
        ax.plot(df['episode'], df['mean_return_10'], alpha=0.8, label=label)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Mean Return (10 episodes)')
    ax.set_title('Training Comparison - Rainbow DQN on Space Invaders')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Plot training progress')
    parser.add_argument(
        '--csv',
        type=str,
        required=True,
        help='Path to CSV log file'
    )
    parser.add_argument(
        '--save',
        type=str,
        default=None,
        help='Path to save plot (if not provided, will display)'
    )
    
    args = parser.parse_args()
    
    plot_training_progress(args.csv, args.save)


if __name__ == '__main__':
    main()
