

from buffers import replay_buffer
import torch
import numpy as np
from numpy.random import uniform, randint
from collections import deque
import time
from datetime import timedelta
import os
import csv
from RainbowAgent import RainbowDQN
from AtariWrapper import make_atari_env
from buffers.replay_buffer import PrioritisedReplayBuffer
from buffers.n_step_buffer import NStepBuffer
import copy

from configs.pbtConfigs.SpaceInvadersConfig import (
    ENV_CONFIG,
    PBT_AGENTS_CONFIG_TYPE,
    STABLE_AGENT_CONFIG,
    BUFFER_CONFIG,
    TRAINING_CONFIG,
    LOGGING_CONFIG,
    DEVICE,
    SEED,
    PBT_CONFIG,
    PBT_AGENTS_CONFIG
)

from scripts.evaluation import eval as evaluate_agent
from logs.pbt_logger import PBTLogger

import multiprocessing as mp
from torch.multiprocessing import Process, Queue, Manager


chekcpoint_path = LOGGING_CONFIG['checkpoint_dir']
logs_path = LOGGING_CONFIG['checkpoint_dir']
peturb_fraction = PBT_CONFIG['perturb_fraction']

class Member():
    def __init__(self, id, device = DEVICE, agent_config = PBT_AGENTS_CONFIG, unstable_types = PBT_AGENTS_CONFIG_TYPE, stable_config = STABLE_AGENT_CONFIG):
        self.id = id
        self.score = -float('inf')  # Initialize with worst fitness
        self.config = {}

        self.agent_config = agent_config
        self.unstable_types = unstable_types
        self.device = device

        # Initialize logger
        self.logger = PBTLogger(member_id=id, base_dir=logs_path)
        
        # Initialize random config for unstable parameters
        for key in agent_config['lower_bounds'].keys():
            if unstable_types[key]=='int': 
                self.config[key] = randint(
                    agent_config['lower_bounds'][key],
                    agent_config['upper_bounds'][key]+1
                )
            elif unstable_types[key]=='float':
                self.config[key] = uniform(
                    agent_config['lower_bounds'][key],
                    agent_config['upper_bounds'][key]
                )

        # Add stable config
        for key in stable_config.keys():
            self.config[key] = stable_config[key]

        # Log initial configuration
        self.logger.log_init_config(self.config)

        self.replay_buffer = PrioritisedReplayBuffer(config_dict=BUFFER_CONFIG, alpha=self.config['alpha'], beta_start=self.config['beta_start'])
        self.n_step_buffer = NStepBuffer(n_step=self.config['n_step'], gamma=self.config['gamma'])
        self.agent = RainbowDQN(self.config, device=device)

    def evaluate(self, eval_env, seed, episodes=5):
        eval_reward, eval_actions, eval_rewards_eval, eval_states = evaluate_agent(eval_env, self.agent, seed, episodes)
        self.score = eval_reward
        return self.score, eval_actions, eval_rewards_eval, eval_states

    def exploit(self, better_config, better_params, better_id, episode=0, total_steps=0):
        """Copy config and weights from better performing member"""
        old_config = copy.deepcopy(self.config)
        
        self.config = copy.deepcopy(better_config)
        self.agent = RainbowDQN(self.config, self.device)
        self.agent.load_state_dict(better_params)
        self.replay_buffer = PrioritisedReplayBuffer(config_dict=BUFFER_CONFIG, alpha=self.config['alpha'], beta_start=self.config['beta_start'])
        self.n_step_buffer = NStepBuffer(n_step=self.config['n_step'], gamma=self.config['gamma'])
        
        # Log exploit event
        self.logger.log_exploit(
            source_member=better_id,
            episode=episode,
            total_steps=self.agent.learn_step_counter,
            old_config=old_config,
            new_config=self.config
        )

    def explore(self, unstable_types=None, agent_config=None, episode=0, total_steps=0):
        """Randomly perturb hyperparameters"""
        if unstable_types is None:
            unstable_types = self.unstable_types
        
        if agent_config is None:
            agent_config = self.agent_config
        
        old_config = copy.deepcopy(self.config)
        
        for key in unstable_types.keys():
            if uniform(0, 1) < peturb_fraction:
                if unstable_types[key]=='int': 
                    self.config[key] = randint(
                        agent_config['lower_bounds'][key],
                        agent_config['upper_bounds'][key]+1
                    )
                elif unstable_types[key]=='float':
                    self.config[key] = uniform(
                        agent_config['lower_bounds'][key],
                        agent_config['upper_bounds'][key]
                    )
        state_dict = self.agent.state_dict()
        self.agent = RainbowDQN(self.config, self.device)
        self.n_step_buffer.gamma = self.config['gamma']
        self.agent.load_state_dict(state_dict)
        
        # Log explore event
        self.logger.log_explore(
            episode=episode,
            total_steps=self.agent.learn_step_counter,
            old_config=old_config,
            new_config=self.config
        )
