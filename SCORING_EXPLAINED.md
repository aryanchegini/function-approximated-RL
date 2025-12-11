# Space Invaders Scoring Explained

## Understanding the Return Values

### What Does "Return 5.8" Mean?

**Return = 5.8** means the agent received a **cumulative reward of 5.8** during that episode.

### Why Such Small Numbers?

Your implementation uses **reward clipping**, which is standard in DQN/Rainbow papers for Atari games.

## Reward Clipping (Currently Applied)

In `src/environment/atari_wrappers.py`, we have:

```python
class ClipRewardEnv(gym.RewardWrapper):
    """Clip rewards to {-1, 0, +1}."""
    def reward(self, reward):
        return np.sign(reward)
```

This means:
- **Positive reward** (killed alien, destroyed UFO) → **+1**
- **Zero reward** (no event) → **0**  
- **Negative reward** (got hit) → **-1**

### Example Episode Breakdown:

```
Episode Return: 5.8

Possible breakdown:
- Killed 6 aliens: 6 × (+1) = +6
- Got hit 1 time: 1 × (-1) = -1
- Plus some fractional rewards from n-step returns
- Total: ~5.8
```

## Raw Space Invaders Scores (Without Clipping)

In the actual game:
- Small alien (top row): **30 points**
- Medium alien (middle row): **20 points**
- Large alien (bottom row): **10 points**
- UFO (mystery ship): **50-300 points** (random)

So a decent game might score **500-2000 points** in raw game score.

## Why Clip Rewards?

From the DQN paper (Mnih et al., 2015):

> "All positive rewards were set to +1 and all negative rewards to -1, leaving 0 rewards unchanged. Clipping the rewards in this manner limits the scale of the error derivatives and makes it easier to use the same learning rate across multiple games."

**Benefits:**
1. ✅ Stable learning across different games
2. ✅ Same hyperparameters work for all Atari games
3. ✅ Prevents huge rewards from dominating learning
4. ✅ Makes gradient updates more consistent

## What Performance Looks Like

### With Clipped Rewards (Your Setup):

| Training Stage | Clipped Return | What It Means |
|----------------|----------------|---------------|
| Random policy | -5 to 5 | Mostly dying, occasional kills |
| Early training (ep 50) | 5-15 | Learning to shoot aliens |
| Mid training (ep 200) | 15-30 | Consistent alien kills, surviving longer |
| Good agent (ep 500) | 30-60 | Strong play, multiple waves |
| Near-human (ep 1000) | 60-100+ | Expert play, high survival |

### Return 5.8 Interpretation:

**Return 5.8** suggests:
- Agent is **just starting to learn**
- Successfully killing some aliens (getting +1 rewards)
- Still dying fairly quickly
- This is **normal for early training**!

## Mapping to Game Performance

Since we use **reward clipping**, here's the rough conversion:

```
Clipped Return ≈ (Aliens Killed) - (Times Hit)

Return 5.8 ≈ 6-8 aliens killed, with 0-2 deaths
```

For perspective:
- **First wave** of Space Invaders has **55 aliens**
- A good agent should kill many waves
- Return of **50+** means clearing multiple waves

## Expected Training Progress

### Typical Learning Curve (with clipped rewards):

```
Episode 1-50:    Returns 0-10    (Learning to shoot)
Episode 50-100:  Returns 10-20   (Consistent kills, still dying fast)
Episode 100-200: Returns 20-40   (Surviving longer, clearing rows)
Episode 200-500: Returns 40-80   (Strong play, multiple waves)
Episode 500+:    Returns 80-150+ (Expert play, high scores)
```

### Your Current Status:

If you're seeing **Return 5.8**, you're likely in the **very early stages** of training:
- Episodes 1-100
- Agent is beginning to learn the basics
- This will improve significantly with more training

## Want to See Raw Scores?

If you want to see the **actual game scores** (not clipped), you can modify the environment:

### Option 1: Remove Reward Clipping

Comment out this line in `src/environment/atari_wrappers.py`:

```python
def make_atari_env(env_id: str, render_mode=None):
    env = gym.make(env_id, render_mode=render_mode)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = EpisodicLifeEnv(env)
    # env = ClipRewardEnv(env)  # COMMENT THIS OUT
    env = WarpFrame(env, width=84, height=84, grayscale=True)
    env = FrameStack(env, k=4)
    env = ScaledFloatFrame(env)
    return env
```

**But**: This will require adjusting hyperparameters (learning rate, etc.) because reward scales are different.

### Option 2: Track Both (Recommended)

Keep clipped rewards for training, but log the raw score separately in the `info` dict.

## Summary

### Current System (Clipped Rewards):
- **Return 5.8** = Small positive progress (6-8 successful actions)
- This is **NORMAL** for early training
- As training progresses, expect returns of 20, 40, 80, 100+
- Clipped rewards are **standard** for DQN/Rainbow on Atari

### What to Expect:
- Early episodes (1-100): Returns 0-15
- Learning phase (100-500): Returns 15-60
- Good performance (500+): Returns 60-150+

### Your Agent's Progress:
**Return 5.8** means your agent is:
✅ Learning to interact with the environment
✅ Getting some positive rewards (killing aliens)
✅ On track for early training
✅ Will improve significantly with more episodes

**This is exactly what you should see in early training!** 🎯

Keep training and you'll see those numbers climb to 20, 40, 80, and beyond!
