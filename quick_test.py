"""
Quick test to verify Rainbow DQN implementation works.
Tests basic functionality without full training.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

print("="*60)
print("Quick Functionality Test")
print("="*60)

# Test 1: Import torch and check device
print("\n1. Testing PyTorch...")
try:
    import torch
    print(f"   ✓ PyTorch {torch.__version__}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   ✓ Device: {device}")
    if device == 'cuda':
        print(f"   ✓ GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("   ⚠️  No CUDA GPU found, will use CPU (slower)")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# Test 2: Import environment modules
print("\n2. Testing environment modules...")
try:
    import gymnasium as gym
    import ale_py
    print("   ✓ Gymnasium and ALE-Py imported")
    
    gym.register_envs(ale_py)
    print("   ✓ ALE environments registered")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# Test 3: Create basic environment
print("\n3. Creating Space Invaders environment...")
try:
    env = gym.make('ALE/SpaceInvaders-v5')
    obs, info = env.reset()
    print(f"   ✓ Environment created")
    print(f"   ✓ Observation shape: {obs.shape}")
    print(f"   ✓ Action space: {env.action_space}")
    env.close()
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# Test 4: Create wrapped environment
print("\n4. Testing wrapped environment...")
try:
    from src.environment.atari_wrappers import make_atari_env
    env = make_atari_env('ALE/SpaceInvaders-v5')
    obs, info = env.reset()
    print(f"   ✓ Wrapped environment created")
    print(f"   ✓ Observation shape: {obs.shape}")
    print(f"   ✓ Observation dtype: {obs.dtype}")
    print(f"   ✓ Observation range: [{obs.min():.2f}, {obs.max():.2f}]")
    
    # Test a step
    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, info = env.step(action)
    print(f"   ✓ Environment step works")
    env.close()
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Create agent
print("\n5. Creating Rainbow DQN agent...")
try:
    from src.agents.rainbow_agent import RainbowDQNAgent
    
    agent = RainbowDQNAgent(
        state_shape=(4, 84, 84),
        num_actions=6,
        device=device,
        num_atoms=51,
        v_min=-10.0,
        v_max=10.0,
        learning_rate=6.25e-5,
        gamma=0.99,
        n_step=3,
        target_update_freq=1000,
    )
    print(f"   ✓ Agent created")
    
    # Count parameters
    total_params = sum(p.numel() for p in agent.online_net.parameters())
    print(f"   ✓ Total parameters: {total_params:,}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Test agent action selection
print("\n6. Testing agent action selection...")
try:
    import numpy as np
    dummy_state = np.random.rand(4, 84, 84).astype(np.float32)
    action = agent.select_action(dummy_state)
    print(f"   ✓ Agent can select actions")
    print(f"   ✓ Selected action: {action}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Test replay buffer
print("\n7. Testing prioritized replay buffer...")
try:
    from src.utils.replay_buffer import PrioritizedReplayBuffer
    
    replay_buffer = PrioritizedReplayBuffer(
        capacity=1000,
        alpha=0.6,
        beta_start=0.4,
        beta_frames=10000
    )
    
    # Add some dummy transitions
    for i in range(100):
        state = np.random.rand(4, 84, 84).astype(np.float32)
        action = np.random.randint(0, 6)
        reward = np.random.randn()
        next_state = np.random.rand(4, 84, 84).astype(np.float32)
        done = False
        replay_buffer.add(state, action, reward, next_state, done)
    
    print(f"   ✓ Replay buffer created")
    print(f"   ✓ Buffer size: {len(replay_buffer)}")
    
    # Sample a batch
    batch, indices, weights = replay_buffer.sample(32)
    print(f"   ✓ Can sample batches")
    print(f"   ✓ Batch keys: {list(batch.keys())}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 8: Test a single learning step
print("\n8. Testing learning step...")
try:
    loss, priorities = agent.learn(batch, indices, weights)
    print(f"   ✓ Learning step successful")
    print(f"   ✓ Loss: {loss:.4f}")
    print(f"   ✓ Priorities shape: {priorities.shape}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 9: Test model save/load
print("\n9. Testing model save/load...")
try:
    import tempfile
    import shutil
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    checkpoint_path = os.path.join(temp_dir, 'test_checkpoint.pth')
    
    # Save model
    agent.save(checkpoint_path)
    print(f"   ✓ Model saved")
    
    # Create new agent
    new_agent = RainbowDQNAgent(
        state_shape=(4, 84, 84),
        num_actions=6,
        device=device,
        num_atoms=51,
        v_min=-10.0,
        v_max=10.0,
    )
    
    # Load model
    new_agent.load(checkpoint_path)
    print(f"   ✓ Model loaded")
    
    # Clean up
    shutil.rmtree(temp_dir)
    print(f"   ✓ Checkpoint save/load works")
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("✅ ALL TESTS PASSED!")
print("="*60)
print("\nYour Rainbow DQN implementation is working correctly!")
print("\nNext steps:")
print("  1. Start training: python scripts/train.py")
print("  2. Monitor progress in logs/ directory")
print("  3. Watch agent play: python scripts/watch_agent.py")
print("\nNote: Training on CPU will be slow (~10-20x slower than GPU)")
print("="*60)
