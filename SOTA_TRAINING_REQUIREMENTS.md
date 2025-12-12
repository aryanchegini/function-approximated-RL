# State-of-the-Art Results for Rainbow DQN on Space Invaders

## Quick Answer:

To get **state-of-the-art results** with Rainbow DQN on Space Invaders, you typically need:

🎯 **10-50 Million timesteps** (10M - 50M steps)

## Timeline Breakdown:

| Training Stage | Timesteps | Performance | What to Expect |
|----------------|-----------|-------------|----------------|
| **Random baseline** | 0 | ~150-200 raw score | Agent shoots randomly |
| **Early learning** | 100K - 500K | ~300-500 raw score | Basic shooting, some dodging |
| **Competent play** | 1M - 5M | ~800-1500 raw score | Consistent alien kills, surviving waves |
| **Strong performance** | 5M - 10M | ~1500-2500 raw score | Multiple waves, strategic play |
| **Near state-of-art** | 10M - 20M | ~2500-4000 raw score | Expert-level play |
| **State-of-the-art** | 20M - 50M | ~3000-5000+ raw score | Peak Rainbow DQN performance |

## Published Benchmark Results:

### Original Papers:

1. **DQN (Nature, 2015)**:
   - Training: 200 million frames (50M timesteps with frame skip=4)
   - Space Invaders score: **1,976** (human normalized: ~200%)
   
2. **Rainbow DQN (AAAI, 2018)**:
   - Training: 200 million frames (50M timesteps)
   - Space Invaders score: **18,789** (human normalized: ~2700%!)
   - This is one of the **best published results**

3. **Human Expert Baseline**:
   - Average score: **1,652**
   - Rainbow significantly exceeds human performance

### Reference Table from Rainbow Paper:

| Algorithm | Space Invaders Score | Training Frames |
|-----------|---------------------|-----------------|
| Random | ~150 | N/A |
| Human | 1,652 | N/A |
| DQN | 1,976 | 200M |
| Double DQN | 3,154 | 200M |
| Prioritized DQN | 1,075 | 200M |
| Dueling DQN | 3,154 | 200M |
| A3C | 1,034 | 320M |
| Distributional DQN (C51) | 2,020 | 200M |
| Noisy DQN | 2,305 | 200M |
| **Rainbow (all combined)** | **18,789** | 200M |

## Your Current Training:

### Current Setup:
```python
TRAINING_CONFIG = {
    'num_episodes': 1000,
    'max_steps_per_episode': 10000,
}
```

### Estimated Timesteps:

If each episode averages ~50-100 steps early on, then longer as agent improves:

```
1000 episodes × ~100 steps average = ~100,000 timesteps (0.1M)
```

**This is early-stage training!** You're seeing scores around 5-7 (with reward clipping), which translates to ~50-100 raw score.

## How to Get State-of-the-Art Results:

### Option 1: Train MUCH Longer (Recommended)

Change your config to train for more episodes:

```python
TRAINING_CONFIG = {
    'num_episodes': 50000,  # Instead of 1000
    'max_steps_per_episode': 10000,
}
```

**Estimated training time** (on CPU):
- 1000 episodes: ~2-4 hours
- 10,000 episodes: ~20-40 hours (1-2 days)
- 50,000 episodes: ~100-200 hours (4-8 days)
- For 50M timesteps: Could take weeks on CPU!

### Option 2: Use GPU (Strongly Recommended)

Rainbow DQN is computationally intensive. For state-of-the-art results:

- **CPU training**: Weeks to months for 50M steps
- **GPU training (CUDA)**: 1-3 days for 50M steps

### Option 3: Realistic Targets for Your Setup

Given you're on CPU, here are more realistic milestones:

| Target | Episodes | Timesteps | Expected Score | Training Time (CPU) |
|--------|----------|-----------|----------------|---------------------|
| **Current** | 1,000 | ~100K | 100-200 | 2-4 hours |
| **Decent** | 5,000 | ~500K | 300-500 | 10-20 hours |
| **Good** | 10,000 | ~1-2M | 500-1000 | 20-40 hours |
| **Strong** | 25,000 | ~3-5M | 1000-1500 | 50-100 hours |
| **Near SOTA** | 100,000+ | ~10-20M | 2000-3000 | 200-400 hours |
| **State-of-art** | 200,000+ | ~50M | 3000-5000+ | Weeks/months |

## Realistic Expectations:

### For Your Current Setup (1000 episodes, CPU):

✅ **What you WILL achieve**:
- Basic competence: Shooting aliens consistently
- Some dodging behavior
- Scores of 200-500 (raw game score)
- Understanding of how Rainbow DQN works

❌ **What you WON'T achieve** (yet):
- State-of-the-art scores (3000+)
- Expert-level strategic play
- Clearing many waves consistently

### Recommendations:

#### For Learning/Coursework (Current Approach):
```python
TRAINING_CONFIG = {
    'num_episodes': 5000,  # ~500K timesteps
    'eval_frequency': 100,
}
```
- Training time: ~12-24 hours (overnight + next day)
- Expected performance: Decent, demonstrable learning
- Good for understanding Rainbow DQN

#### For Strong Results:
```python
TRAINING_CONFIG = {
    'num_episodes': 20000,  # ~2-5M timesteps
    'eval_frequency': 200,
}
```
- Training time: 2-4 days on CPU
- Expected performance: Good scores, multiple waves
- Impressive for a coursework project

#### For State-of-the-Art:
```python
TRAINING_CONFIG = {
    'num_episodes': 100000,  # ~10-20M timesteps
    'eval_frequency': 500,
}
```
- Training time: Weeks on CPU, days on GPU
- Expected performance: Near-human or superhuman
- Requires GPU for practical training

## Current Progress Check:

Your training so far:
```
Episode 500: Total Steps ~29,357
Eval return: 6.4 (clipped) ≈ ~100-150 raw score
```

You're at **~30K timesteps** out of 50M needed for SOTA. That's:
- **0.06%** of the way to state-of-the-art
- But showing **good learning progress** for this stage!

## Learning Curve (Typical):

```
Timesteps     | Space Invaders Score | Performance Level
--------------|----------------------|-------------------
0             | 150-200             | Random
10K - 50K     | 200-300             | ← You are here
50K - 200K    | 300-500             | Basic competence
200K - 1M     | 500-1000            | Decent play
1M - 5M       | 1000-2000           | Good play
5M - 10M      | 2000-3000           | Strong play
10M - 20M     | 2500-4000           | Near SOTA
20M - 50M     | 3000-5000+          | State-of-the-art
```

## Important Note: Reward Clipping

Remember, your current implementation uses **reward clipping** (standard for Rainbow):

- **Clipped return of 6.4** ≈ roughly **100-150 raw game score**
- For comparison: State-of-art Rainbow gets **18,789 raw score**
- That would be clipped to ~150-300 return (since each positive event = +1)

To see actual game scores, you'd need to remove reward clipping (but this would require retuning hyperparameters).

## Summary:

### Your Question: "How many timesteps for SOTA?"
**Answer: 20-50 million timesteps**

### Your Current Status:
- **~30K timesteps** after 500 episodes
- **Expected with 1000 episodes**: ~100K timesteps
- **This is 0.2-0.5%** of the way to SOTA

### Realistic Path Forward:

1. **For coursework** (demonstrate learning): 
   - Train 5,000-10,000 episodes (~500K-1M steps)
   - Achieves decent performance in 12-48 hours

2. **For impressive results** (multiple waves):
   - Train 20,000-50,000 episodes (~2-5M steps)
   - Requires 2-7 days on CPU

3. **For state-of-the-art** (superhuman play):
   - Train 100,000+ episodes (~20-50M steps)
   - Requires GPU or weeks of CPU time

### Bottom Line:

Your current 1000-episode training will show **clear learning** but won't reach state-of-the-art. For coursework purposes, this is probably fine! If you want stronger results, increase to 5000-10000 episodes for a good balance between training time and performance.

**Rainbow DQN achieved its famous results after 200M frames = 50M timesteps = weeks of GPU training!** 🚀
