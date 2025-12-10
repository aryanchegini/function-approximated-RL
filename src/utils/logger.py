"""
Logging utilities for training.
"""
import os
import csv
from typing import Dict, Any
from datetime import datetime


class Logger:
    """Logger for training metrics."""
    
    def __init__(self, log_dir: str, experiment_name: str):
        """
        Args:
            log_dir: Directory to save logs
            experiment_name: Name of the experiment
        """
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # CSV file path
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.csv_path = os.path.join(
            log_dir,
            f'{experiment_name}_{timestamp}.csv'
        )
        
        # Initialize CSV
        self.csv_initialized = False
        self.fieldnames = []
    
    def log_episode(self, episode: int, metrics: Dict[str, Any]):
        """
        Log episode metrics.
        
        Args:
            episode: Episode number
            metrics: Dictionary of metrics to log
        """
        # Add episode number
        data = {'episode': episode, **metrics}
        
        # Initialize CSV on first call
        if not self.csv_initialized:
            self.fieldnames = list(data.keys())
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()
            self.csv_initialized = True
        
        # Append data
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(data)
    
    def print_episode(self, episode: int, metrics: Dict[str, Any]):
        """Print episode metrics to console."""
        msg = f"Episode {episode}"
        for key, value in metrics.items():
            if isinstance(value, float):
                msg += f" | {key}: {value:.2f}"
            else:
                msg += f" | {key}: {value}"
        print(msg)
    
    def get_csv_path(self) -> str:
        """Get path to CSV log file."""
        return self.csv_path


class TensorBoardLogger:
    """TensorBoard logger wrapper."""
    
    def __init__(self, log_dir: str, experiment_name: str):
        """
        Args:
            log_dir: Directory to save logs
            experiment_name: Name of the experiment
        """
        try:
            from torch.utils.tensorboard import SummaryWriter
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            tensorboard_dir = os.path.join(log_dir, 'tensorboard', f'{experiment_name}_{timestamp}')
            self.writer = SummaryWriter(tensorboard_dir)
            self.enabled = True
            print(f"TensorBoard logging enabled. Run: tensorboard --logdir={tensorboard_dir}")
        except ImportError:
            print("TensorBoard not available. Install with: pip install tensorboard")
            self.enabled = False
            self.writer = None
    
    def log_scalar(self, tag: str, value: float, step: int):
        """Log scalar value."""
        if self.enabled:
            self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int):
        """Log multiple scalar values."""
        if self.enabled:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)
    
    def close(self):
        """Close TensorBoard writer."""
        if self.enabled:
            self.writer.close()
