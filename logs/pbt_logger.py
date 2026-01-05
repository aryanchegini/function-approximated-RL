import os
import csv
import json
from datetime import datetime
from typing import Dict, Any, Optional
import copy


class PBTLogger:
    """
    Handles episode logs, exploration logs, and checkpoints.
    """
    
    def __init__(self, member_id: int, base_dir: str):
        """
        Args:
            member_id: Member ID
            base_dir: Base checkpoint directory (e.g., 'pbt_checkpoints')
        """
        self.member_id = member_id
        self.base_dir = base_dir
        
        # Create member directory
        self.member_dir = os.path.join(base_dir, f'member_{member_id}')
        os.makedirs(self.member_dir, exist_ok=True)
        
        # Log file paths
        self.episode_log = os.path.join(self.member_dir, 'episode_log.csv')
        self.exploration_log = os.path.join(self.member_dir, 'exploration_log.csv')
        
        # Config directory
        self.config_dir = os.path.join(self.member_dir, 'configs')
        os.makedirs(self.config_dir, exist_ok=True)
        
        # Initialize CSV files
        self._init_episode_log()
        self._init_exploration_log()
    
    def _init_episode_log(self):
        with open(self.episode_log, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp',
                'episode',
                'total_steps',
                'episode_return',
                'episode_length',
                'mean_return_10',
                'mean_return_100',
                'avg_loss',
                'buffer_size'
            ])
    
    def _init_exploration_log(self):
        with open(self.exploration_log, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp',
                'change_type',
                'source_member',
                'episode',
                'total_steps',
                'param_name',
                'old_value',
                'new_value'
            ])
    
    @staticmethod
    def _get_timestamp():
        return datetime.now().isoformat()
    
    def log_episode(self, 
                   episode: int,
                   total_steps: int,
                   episode_return: float,
                   episode_length: int,
                   mean_return_10: float,
                   mean_return_100: float,
                   avg_loss: float,
                   buffer_size: int):
        """
        Log episode metrics to CSV.
        """
        timestamp = self._get_timestamp()
        
        with open(self.episode_log, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                episode,
                total_steps,
                episode_return,
                episode_length,
                mean_return_10,
                mean_return_100,
                avg_loss,
                buffer_size
            ])
            f.flush()
    
    def log_console(self,
                   episode: int,
                   max_episodes: int,
                   episode_steps: int,
                   episode_reward: float,
                   mean_return_100: float,
                   total_steps: int,
                   avg_loss: float,
                   buffer_size: int,
                   time_str: str):
        """
        Print formatted console output for training progress.
        """
        print(f" Agent {self.member_id} | Episode {episode}/{max_episodes} | "
              f"Steps: {episode_steps} | "
              f"Reward: {episode_reward:.2f} | "
              f"Mean(100): {mean_return_100:.1f} | "
              f"Total Steps: {total_steps} | "
              f"Avg Loss: {avg_loss:.4f} | "
              f"Buffer: {buffer_size} | "
              f"Time: {time_str}")
    
    def log_init_config(self, config: Dict[str, Any]):
        timestamp = self._get_timestamp()
        
        # Save full config as JSON
        config_path = os.path.join(self.config_dir, 'config_init.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Log each parameter as 'init' type
        with open(self.exploration_log, 'a', newline='') as f:
            writer = csv.writer(f)
            for param_name, value in config.items():
                writer.writerow([
                    timestamp,
                    'init',
                    '',  # No source member
                    0,   # Episode 0
                    0,   # Total steps 0
                    param_name,
                    '',  # No old value
                    value
                ])
            f.flush()
    
    def log_exploit(self,
                   source_member: int,
                   episode: int,
                   total_steps: int,
                   old_config: Dict[str, Any],
                   new_config: Dict[str, Any]):
        """
        Log exploit event (copying from better member).
        """
        timestamp = self._get_timestamp()
        
        # Save config snapshot
        config_path = os.path.join(
            self.config_dir,
            f'config_exploit_ep{episode}_step{total_steps}.json'
        )
        with open(config_path, 'w') as f:
            json.dump(new_config, f, indent=2)
        
        # Log only changed parameters
        with open(self.exploration_log, 'a', newline='') as f:
            writer = csv.writer(f)
            for param_name in new_config.keys():
                old_val = old_config.get(param_name, '')
                new_val = new_config[param_name]
                if old_val != new_val:
                    writer.writerow([
                        timestamp,
                        'exploit',
                        source_member,
                        episode,
                        total_steps,
                        param_name,
                        old_val,
                        new_val
                    ])
            f.flush()
    
    def log_explore(self,
                   episode: int,
                   total_steps: int,
                   old_config: Dict[str, Any],
                   new_config: Dict[str, Any]):
        """
        Log explore event (random perturbation).
        """
        timestamp = self._get_timestamp()
        
        # Save config snapshot
        config_path = os.path.join(
            self.config_dir,
            f'config_explore_ep{episode}_step{total_steps}.json'
        )
        with open(config_path, 'w') as f:
            json.dump(new_config, f, indent=2)
        
        # Log only changed parameters
        with open(self.exploration_log, 'a', newline='') as f:
            writer = csv.writer(f)
            for param_name in new_config.keys():
                old_val = old_config.get(param_name, '')
                new_val = new_config[param_name]
                if old_val != new_val:
                    writer.writerow([
                        timestamp,
                        'explore',
                        '',  # No source member
                        episode,
                        total_steps,
                        param_name,
                        old_val,
                        new_val
                    ])
            f.flush()


class GlobalCheckpointManager:
    """
    Manages global best model and periodic checkpoints.
    Shared across all members.
    """
    
    def __init__(self,
                 base_dir: str,
                 num_checkpoints: int = 10,
                 total_steps: Optional[int] = None,
                 checkpoint_by: str = 'steps'):
        """
        Args:
            base_dir: Base checkpoint directory
            num_checkpoints: Number of periodic checkpoints (default 10)
            total_steps: Total training steps
            checkpoint_by: 'steps' or 'episodes'
        """
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        
        # Global best tracking
        self.best_score = -float('inf')
        self.best_model_path = os.path.join(base_dir, 'best_model.pt')
        self.best_metadata_path = os.path.join(base_dir, 'best_metadata.json')
        
        # Periodic checkpoints
        self.checkpoint_dir = os.path.join(base_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.checkpoint_log = os.path.join(base_dir, 'checkpoints_log.csv')
        self.num_checkpoints = num_checkpoints
        self.checkpoint_by = checkpoint_by
        
        # Calculate intervals
        if total_steps and num_checkpoints > 0:
            self.checkpoint_interval = total_steps / num_checkpoints
        else:
            self.checkpoint_interval = None
        
        self.next_checkpoint_num = 1
        self.next_checkpoint_threshold = self.checkpoint_interval if self.checkpoint_interval else float('inf')
        
        self._init_checkpoint_log()
    
    def _init_checkpoint_log(self):
        if not os.path.exists(self.checkpoint_log):
            with open(self.checkpoint_log, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'checkpoint_num',
                    'timestamp',
                    'member_id',
                    'episode',
                    'total_steps',
                    'eval_score',
                    'model_path'
                ])
    
    def update_best(self,
                   member_id: int,
                   episode: int,
                   total_steps: int,
                   score: float,
                   agent,
                   config: Dict[str, Any]) -> bool:
        """
        Update global best model if score is better.
        
        Returns:
            True if new best was saved
        """
        if score > self.best_score:
            self.best_score = score
            
            # Save model
            agent.save(self.best_model_path)
            
            # Save metadata
            metadata = {
                'member_id': member_id,
                'episode': episode,
                'total_steps': total_steps,
                'score': score,
                'timestamp': datetime.now().isoformat(),
                'config': config
            }
            with open(self.best_metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return True
        return False
    
    def should_save_checkpoint(self, current_steps: int) -> bool:
        return (self.checkpoint_interval is not None and
                current_steps >= self.next_checkpoint_threshold and
                self.next_checkpoint_num <= self.num_checkpoints)
    
    def save_checkpoint(self,
                       member_id: int,
                       episode: int,
                       total_steps: int,
                       score: float,
                       agent,
                       config: Dict[str, Any]):
        """Save a periodic checkpoint"""
        checkpoint_num = self.next_checkpoint_num
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f'checkpoint_{checkpoint_num:02d}.pt'
        )
        
        # Save model
        agent.save(checkpoint_path)
        
        # Log checkpoint
        timestamp = datetime.now().isoformat()
        with open(self.checkpoint_log, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                checkpoint_num,
                timestamp,
                member_id,
                episode,
                total_steps,
                score,
                checkpoint_path
            ])
            f.flush()
        
        # Update for next checkpoint
        self.next_checkpoint_num += 1
        self.next_checkpoint_threshold += self.checkpoint_interval