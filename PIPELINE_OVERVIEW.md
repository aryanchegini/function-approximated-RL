# Rainbow DQN Training Pipeline Overview

## 🔄 Complete Training Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                         TRAINING PIPELINE                            │
└─────────────────────────────────────────────────────────────────────┘

1. ENVIRONMENT SETUP
   ├── Space Invaders (ALE/SpaceInvaders-v5)
   ├── Atari Wrappers Applied:
   │   ├── NoOp Reset (random initial states)
   │   ├── Max & Skip (4 frame skip, max pool)
   │   ├── Fire Reset (start game)
   │   ├── Episodic Life (life = episode)
   │   ├── Reward Clipping (to {-1, 0, +1})
   │   ├── Warp Frame (84x84 grayscale)
   │   ├── Frame Stack (stack 4 frames)
   │   └── Float Scale (normalize to [0, 1])
   └── Output: 4x84x84 float tensor

2. AGENT INITIALIZATION
   ├── Rainbow Network (Online)
   │   ├── Conv Layer 1: 4 -> 32 channels
   │   ├── Conv Layer 2: 32 -> 64 channels
   │   ├── Conv Layer 3: 64 -> 64 channels
   │   ├── Value Stream: 64*7*7 -> 512 -> 51 atoms
   │   └── Advantage Stream: 64*7*7 -> 512 -> (6 actions × 51 atoms)
   ├── Rainbow Network (Target)
   │   └── Copy of online network (updated every 1000 steps)
   └── Optimizer: Adam (lr=6.25e-5)

3. MEMORY INITIALIZATION
   ├── Prioritized Replay Buffer
   │   ├── Capacity: 100,000 transitions
   │   ├── Sum Tree structure
   │   ├── Alpha: 0.6 (prioritization)
   │   └── Beta: 0.4 -> 1.0 (importance sampling)
   └── N-Step Buffer
       ├── N: 3 steps
       └── Gamma: 0.99

4. TRAINING LOOP (per episode)
   ┌─────────────────────────────────────────────┐
   │  Episode Loop (1000 episodes)               │
   │                                             │
   │  ┌───────────────────────────────────────┐ │
   │  │ Step Loop (max 10,000 steps)          │ │
   │  │                                       │ │
   │  │ 1. Observe state: (4, 84, 84)        │ │
   │  │                                       │ │
   │  │ 2. Select action:                    │ │
   │  │    - Forward pass through network    │ │
   │  │    - Noisy networks (no ε-greedy)    │ │
   │  │    - Greedy w.r.t Q-values           │ │
   │  │                                       │ │
   │  │ 3. Execute action in environment     │ │
   │  │                                       │ │
   │  │ 4. Store in N-step buffer            │ │
   │  │    - Accumulate n-step return        │ │
   │  │    - When full, add to replay buffer │ │
   │  │                                       │ │
   │  │ 5. Sample batch (if ready):          │ │
   │  │    ├── Sample 32 transitions         │ │
   │  │    ├── Prioritized sampling          │ │
   │  │    └── Compute IS weights            │ │
   │  │                                       │ │
   │  │ 6. Learn from batch:                 │ │
   │  │    ├── Compute current distribution  │ │
   │  │    ├── Double Q-learning:            │ │
   │  │    │   - Select action with online   │ │
   │  │    │   - Evaluate with target        │ │
   │  │    ├── N-step Bellman update         │ │
   │  │    ├── Categorical projection        │ │
   │  │    ├── Cross-entropy loss            │ │
   │  │    ├── Weighted by IS weights        │ │
   │  │    ├── Backprop + gradient clip      │ │
   │  │    ├── Update priorities             │ │
   │  │    └── Reset noise                   │ │
   │  │                                       │ │
   │  │ 7. Update target network (every      │ │
   │  │    1000 steps)                        │ │
   │  └───────────────────────────────────────┘ │
   │                                             │
   │  Episode End:                               │
   │  ├── Log metrics to CSV                     │
   │  ├── Log to TensorBoard                     │
   │  └── Print progress                         │
   │                                             │
   │  Every 50 episodes:                         │
   │  ├── Evaluate agent (5 episodes)            │
   │  └── Save if best performance               │
   │                                             │
   │  Every 100 episodes:                        │
   │  └── Save checkpoint                        │
   └─────────────────────────────────────────────┘

5. OUTPUT
   ├── Checkpoints/
   │   ├── rainbow_space_invaders_best.pth     (best model)
   │   ├── rainbow_space_invaders_final.pth    (final model)
   │   ├── rainbow_space_invaders_ep100.pth    (every 100 ep)
   │   └── ...
   └── Logs/
       ├── rainbow_space_invaders_TIMESTAMP.csv
       └── tensorboard/
           └── events.out.tfevents...
```

## 🧠 Rainbow DQN Components Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                      RAINBOW DQN ARCHITECTURE                        │
└─────────────────────────────────────────────────────────────────────┘

INPUT: State (4, 84, 84)
   │
   ├─→ Conv1 (32 filters, 8x8, stride 4) + ReLU
   │      │
   │      v
   ├─→ Conv2 (64 filters, 4x4, stride 2) + ReLU
   │      │
   │      v
   ├─→ Conv3 (64 filters, 3x3, stride 1) + ReLU
   │      │
   │      v
   │   Flatten (64 * 7 * 7 = 3136)
   │      │
   │      ├──────────────────┬──────────────────┐
   │      │                  │                  │
   │   VALUE STREAM      ADVANTAGE STREAM      │
   │      │                  │                  │
   │      v                  v                  │
   │   Noisy Linear      Noisy Linear          │
   │   (3136 -> 512)     (3136 -> 512)         │
   │      │                  │                  │
   │      v                  v                  │
   │   ReLU              ReLU                   │
   │      │                  │                  │
   │      v                  v                  │
   │   Noisy Linear      Noisy Linear          │
   │   (512 -> 51)       (512 -> 306)          │ [6 actions × 51 atoms]
   │      │                  │                  │
   │      v                  v                  │
   │   Value Dist        Advantage Dist        │
   │   (1, 51)           (6, 51)               │
   │      │                  │                  │
   │      └──────┬───────────┘                  │
   │             │                              │
   │             v                              │
   │       DUELING COMBINE                      │
   │       Q(s,a) = V(s) + [A(s,a) - mean(A)]  │
   │             │                              │
   │             v                              │
   │       Apply Softmax                        │
   │             │                              │
   │             v                              │
OUTPUT: Distribution over values (6, 51)
        For each action: probability distribution over 51 atoms
        spanning [-10, 10]
```

## 🎯 Learning Process

```
┌─────────────────────────────────────────────────────────────────────┐
│                       LEARNING ALGORITHM                             │
└─────────────────────────────────────────────────────────────────────┘

1. Sample Batch (Prioritized)
   └─→ 32 transitions from replay buffer (prioritized by TD error)

2. Current Distribution
   ├─→ Online network: Q_dist(s, a)
   └─→ Select distribution for taken action

3. Target Distribution (Double Q-Learning)
   ├─→ Online network: select best action a' = argmax Q(s', a')
   ├─→ Target network: evaluate Q_dist(s', a')
   └─→ Select distribution for best action

4. N-Step Bellman Update
   └─→ Project: Tz = r + γ^n * z for each atom z

5. Categorical Projection
   ├─→ Clip: Tz ∈ [V_min, V_max]
   ├─→ Compute: b = (Tz - V_min) / Δz
   ├─→ Lower: l = floor(b)
   ├─→ Upper: u = ceil(b)
   └─→ Distribute probability to neighboring atoms

6. Loss Computation
   ├─→ Cross-entropy: -Σ p_target(z) * log(p_current(z))
   └─→ Weight by importance sampling weights

7. Optimization
   ├─→ Compute gradients
   ├─→ Clip gradients (max norm = 10)
   ├─→ Update online network
   └─→ Update priorities in replay buffer

8. Noise & Target Update
   ├─→ Reset noise in all noisy layers
   └─→ Update target network (every 1000 steps)
```

## 📊 Data Flow

```
Environment → Preprocessing → Agent → Action → Environment
     ↓                                   ↑
     └─→ N-Step Buffer → Replay Buffer ──┘
                              ↓
                         Learning
                              ↓
                    Update Online Network
                              ↓
                (Every 1000 steps) Update Target Network
```

## 🎮 Inference (Watching Agent)

```
1. Load checkpoint
2. Set to eval mode (deterministic noisy layers)
3. For each episode:
   ├─→ Reset environment
   ├─→ While not done:
   │   ├─→ Get state
   │   ├─→ Forward pass → Q-values
   │   ├─→ Select argmax action
   │   ├─→ Execute action
   │   └─→ Render frame
   └─→ Episode complete
```

## 📈 Evaluation Pipeline

```
1. Load checkpoint
2. Set to eval mode
3. Run N episodes (e.g., 100)
4. Collect statistics:
   ├─→ Mean return
   ├─→ Std deviation
   ├─→ Min/Max returns
   ├─→ Median return
   └─→ Mean episode length
5. Report results
```

This pipeline represents a complete, production-ready implementation of 
Rainbow DQN with all components working together seamlessly!
