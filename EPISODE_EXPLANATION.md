# Understanding Episodes and Training Progress

## What is an Episode?

**Episode = One Complete Game Playthrough**

✅ **Yes, your understanding is correct!** Each episode is one full run of Space Invaders:

```
Episode 1:  Start → Play → Die → End
Episode 2:  Start → Play → Die → End
Episode 3:  Start → Play → Die → End
...
Episode 50: Start → Play → Die → End
...
Episode 500: Start → Play → Die → End
```

Each episode runs until:
- ❌ Agent dies (loses all 3 lives)
- ✅ Agent clears all aliens (rare early in training!)
- ⏱️ Maximum steps reached (10,000 steps)

## Why Do You See Output Every 50 Episodes?

The agent plays **EVERY single episode** (1, 2, 3, 4, ..., 500), but **we only PRINT** progress every 50 episodes to avoid cluttering your console.

### Here's What Actually Happens:

```
Episode 1:   Agent plays → Learns → No printout
Episode 2:   Agent plays → Learns → No printout
Episode 3:   Agent plays → Learns → No printout
...
Episode 49:  Agent plays → Learns → No printout
Episode 50:  Agent plays → Learns → PRINTOUT + EVALUATION ✅
Episode 51:  Agent plays → Learns → No printout
Episode 52:  Agent plays → Learns → No printout
...
Episode 99:  Agent plays → Learns → No printout
Episode 100: Agent plays → Learns → PRINTOUT + EVALUATION ✅
```

### Configuration:

From `configs/space_invaders_config.py`:

```python
TRAINING_CONFIG = {
    'num_episodes': 1000,      # Train for 1000 complete games
    'eval_frequency': 50,      # Print/evaluate every 50 episodes
    'eval_episodes': 5,        # Run 5 test games for evaluation
    'save_frequency': 100,     # Save checkpoint every 100 episodes
}
```

## Breaking Down Your Output:

```
Episode 50 | Steps: 72 | Score: 0.0 | Total Steps: 2924 | Mean(100): 0.0 | Eval: 5.8
```

Let me explain each part:

| Part | Meaning |
|------|---------|
| **Episode 50** | This is the 50th complete game played |
| **Steps: 72** | Episode 50 lasted 72 game steps before agent died |
| **Score: 0.0** | The return for episode 50 specifically (clipped rewards) |
| **Total Steps: 2924** | Cumulative steps across ALL 50 episodes (episode 1 + 2 + ... + 50) |
| **Mean(100): 0.0** | Average return over last 100 episodes (only 50 so far, so limited data) |
| **Eval: 5.8** | Agent was tested on 5 NEW games, averaged 5.8 return |

## Training Flow Visualization:

```
                    TRAINING EPISODES (Learning Mode)
                    ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
Episodes 1-49:     Play → Store in memory → Learn → Update network
                    (No printout, just learning)

Episode 50:        Play → Store in memory → Learn → Update network
                    ↓
                    EVALUATION TIME!
                    ↓
                    EVALUATION EPISODES (Testing Mode)
                    ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
                    Test Game 1 → Return: 6.0
                    Test Game 2 → Return: 5.0
                    Test Game 3 → Return: 5.5
                    Test Game 4 → Return: 6.5
                    Test Game 5 → Return: 6.0
                    Average: 5.8 ← This is "Eval: 5.8"
                    ↓
                    PRINT SUMMARY ✅
                    "Episode 50 | ... | Eval: 5.8"

Episodes 51-99:    Play → Store in memory → Learn → Update network
                    (No printout, just learning)

Episode 100:       Play → Store in memory → Learn → Update network
                    ↓
                    EVALUATION + PRINT ✅
                    "Episode 100 | ... | Eval: X.X"
```

## Why Evaluation Every 50 Episodes?

### Evaluation = Testing Without Learning

When we reach episode 50, 100, 150, etc., the agent:
1. **Pauses training** (no more learning updates)
2. **Plays 5 test games** to measure current performance
3. **Reports average performance** across those 5 games
4. **Resumes training**

This gives you a **clean measure** of how well the agent has learned, separate from the noisy training episodes.

### Training vs. Evaluation:

| Aspect | Training Episodes | Evaluation Episodes |
|--------|-------------------|---------------------|
| **Purpose** | Learn and improve | Test current skill |
| **Exploration** | Yes (noisy networks) | No (greedy policy) |
| **Network updates** | Yes | No |
| **Frequency** | Every episode | Every 50 episodes |
| **Count** | 1000 total | 5 per evaluation |

## Total Episodes Played:

If training for 1000 episodes with evaluation every 50:

```
Training episodes: 1000 games
Evaluation episodes: (1000 / 50) × 5 = 20 × 5 = 100 extra games

Total games played: 1000 + 100 = 1100 complete Space Invaders games!
```

## Why "Total Steps: 2924"?

This is the **cumulative sum** of all steps across episodes:

```
Episode 1:  50 steps   → Total: 50
Episode 2:  45 steps   → Total: 95
Episode 3:  60 steps   → Total: 155
...
Episode 50: 72 steps   → Total: 2924
```

The agent has taken **2,924 individual actions** across all 50 episodes.

## Key Takeaways:

1. ✅ **Episode = One complete game** (you were right!)
2. 🎮 Agent plays **every single episode** (1, 2, 3, ..., 1000)
3. 📊 We **print progress every 50 episodes** to keep output clean
4. 🧪 **Evaluation** runs 5 separate test games to measure performance
5. 📈 You're seeing episode **50, 100, 150, 200, ...** because `eval_frequency = 50`

## Want Different Print Frequency?

You can change this in `configs/space_invaders_config.py`:

```python
TRAINING_CONFIG = {
    'eval_frequency': 10,  # Print every 10 episodes instead of 50
    # or
    'eval_frequency': 100,  # Print every 100 episodes (less output)
}
```

## Summary:

Your agent is playing 1000 complete games of Space Invaders, but you only see progress reports every 50 games to keep things clean. Between printouts, the agent is constantly playing, learning, and improving!

**Episode 50 = The 50th complete playthrough of the game** 🎯
