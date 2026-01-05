# function-approximated-RL

# Install dependencies
```bash
pip install -r requirements.txt
```


# Usage guide:
We recommend checking the config files to ensure file directories are appropriate for your device.

train.py trains a rainbow DQN agent

trainPBT.py trains a rainbow DQN agent (4 by default)

trainDQN.py trains a standard DQN agent

Use WatchAgent.py [checkpoint_path].pt to show a checkpoint playing

All trainings currently run for 100_000 episodes (roughly 10M steps).

# Papers this work is based on:

Rainbow DQN: https://arxiv.org/abs/1710.02298

Population Based Training: https://arxiv.org/abs/1711.09846

# Here's a glimpse of our results:

<p align="center">
<img src=".\All results\Plots\1_rainbow_vs_best_pbt.png" height = "250" alt="" align=center />
<br><br>
<b>Figure 1.</b> Plot comparison of Population Based Training vs base Rainbow DQN
</p>

<p align="center">
<img src=".\All results\Plots\2_pbt_best_vs_average.png" height = "250" alt="" align=center />
<br><br>
<b>Figure 2.</b> Population Based Training, all four members
</p>

<p align="center">
<img src=".\All results\Plots\3_all_agents_performance.png" height = "250" alt="" align=center />
<br><br>
<b>Figure 3.</b> Comparison of all agent results
</p>
