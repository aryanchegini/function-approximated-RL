"""
Test script to verify installation and setup.
Run this to check if everything is properly configured.
"""
import sys
import os

def test_imports():
    """Test if all required packages are installed."""
    print("Testing imports...")
    
    errors = []
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        print(f"  - CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  - CUDA device: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        errors.append(f"✗ PyTorch: {e}")
    
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
    except ImportError as e:
        errors.append(f"✗ NumPy: {e}")
    
    try:
        import gymnasium
        print(f"✓ Gymnasium {gymnasium.__version__}")
    except ImportError as e:
        errors.append(f"✗ Gymnasium: {e}")
    
    try:
        import ale_py
        print(f"✓ ALE-Py {ale_py.__version__}")
    except ImportError as e:
        errors.append(f"✗ ALE-Py: {e}")
    
    try:
        import cv2
        print(f"✓ OpenCV {cv2.__version__}")
    except ImportError as e:
        errors.append(f"✗ OpenCV: {e}")
    
    try:
        import matplotlib
        print(f"✓ Matplotlib {matplotlib.__version__}")
    except ImportError as e:
        errors.append(f"✗ Matplotlib: {e}")
    
    try:
        import pandas as pd
        print(f"✓ Pandas {pd.__version__}")
    except ImportError as e:
        errors.append(f"✗ Pandas: {e}")
    
    return errors


def test_environment():
    """Test if Atari environment can be created."""
    print("\nTesting environment creation...")
    
    try:
        import gymnasium as gym
        import ale_py
        
        gym.register_envs(ale_py)
        env = gym.make('ALE/SpaceInvaders-v5')
        obs, info = env.reset()
        print(f"✓ Space Invaders environment created")
        print(f"  - Observation shape: {obs.shape}")
        print(f"  - Action space: {env.action_space}")
        env.close()
        return True
    except Exception as e:
        print(f"✗ Environment creation failed: {e}")
        return False


def test_custom_imports():
    """Test if custom modules can be imported."""
    print("\nTesting custom modules...")
    
    sys.path.insert(0, os.path.dirname(__file__))
    
    errors = []
    
    try:
        from src.networks.noisy_linear import NoisyLinear
        print("✓ NoisyLinear")
    except ImportError as e:
        errors.append(f"✗ NoisyLinear: {e}")
    
    try:
        from src.networks.rainbow_network import RainbowDQNNetwork
        print("✓ RainbowDQNNetwork")
    except ImportError as e:
        errors.append(f"✗ RainbowDQNNetwork: {e}")
    
    try:
        from src.agents.rainbow_agent import RainbowDQNAgent
        print("✓ RainbowDQNAgent")
    except ImportError as e:
        errors.append(f"✗ RainbowDQNAgent: {e}")
    
    try:
        from src.utils.replay_buffer import PrioritizedReplayBuffer
        print("✓ PrioritizedReplayBuffer")
    except ImportError as e:
        errors.append(f"✗ PrioritizedReplayBuffer: {e}")
    
    try:
        from src.utils.n_step import NStepBuffer
        print("✓ NStepBuffer")
    except ImportError as e:
        errors.append(f"✗ NStepBuffer: {e}")
    
    try:
        from src.environment.atari_wrappers import make_atari_env
        print("✓ Atari wrappers")
    except ImportError as e:
        errors.append(f"✗ Atari wrappers: {e}")
    
    return errors


def test_wrapped_environment():
    """Test if wrapped Atari environment works."""
    print("\nTesting wrapped environment...")
    
    try:
        sys.path.insert(0, os.path.dirname(__file__))
        from src.environment.atari_wrappers import make_atari_env
        import gymnasium as gym
        import ale_py
        
        gym.register_envs(ale_py)
        env = make_atari_env('ALE/SpaceInvaders-v5')
        obs, info = env.reset()
        print(f"✓ Wrapped environment created")
        print(f"  - Observation shape: {obs.shape}")
        print(f"  - Observation dtype: {obs.dtype}")
        print(f"  - Observation range: [{obs.min():.2f}, {obs.max():.2f}]")
        
        # Test step
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        print(f"✓ Environment step successful")
        print(f"  - Reward: {reward}")
        print(f"  - Terminated: {terminated}")
        print(f"  - Truncated: {truncated}")
        
        env.close()
        return True
    except Exception as e:
        print(f"✗ Wrapped environment failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_agent_creation():
    """Test if Rainbow agent can be created."""
    print("\nTesting agent creation...")
    
    try:
        sys.path.insert(0, os.path.dirname(__file__))
        from src.agents.rainbow_agent import RainbowDQNAgent
        import torch
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
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
        
        print(f"✓ Rainbow agent created successfully")
        print(f"  - Device: {device}")
        print(f"  - Parameters: {sum(p.numel() for p in agent.online_net.parameters()):,}")
        
        # Test forward pass
        import numpy as np
        dummy_state = np.random.rand(4, 84, 84).astype(np.float32)
        action = agent.select_action(dummy_state)
        print(f"✓ Agent can select actions")
        print(f"  - Selected action: {action}")
        
        return True
    except Exception as e:
        print(f"✗ Agent creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*60)
    print("Rainbow DQN - Installation Test")
    print("="*60)
    print()
    
    # Test imports
    import_errors = test_imports()
    
    # Test environment
    env_ok = test_environment()
    
    # Test custom imports
    custom_errors = test_custom_imports()
    
    # Test wrapped environment
    wrapped_ok = test_wrapped_environment()
    
    # Test agent creation
    agent_ok = test_agent_creation()
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    all_ok = True
    
    if import_errors:
        print("\n⚠ Import Issues:")
        for error in import_errors:
            print(f"  {error}")
        all_ok = False
    else:
        print("\n✓ All dependencies installed correctly")
    
    if custom_errors:
        print("\n⚠ Custom Module Issues:")
        for error in custom_errors:
            print(f"  {error}")
        all_ok = False
    else:
        print("✓ All custom modules importable")
    
    if not env_ok:
        print("✗ Environment creation failed")
        all_ok = False
    else:
        print("✓ Environment creation successful")
    
    if not wrapped_ok:
        print("✗ Wrapped environment failed")
        all_ok = False
    else:
        print("✓ Wrapped environment successful")
    
    if not agent_ok:
        print("✗ Agent creation failed")
        all_ok = False
    else:
        print("✓ Agent creation successful")
    
    print("\n" + "="*60)
    if all_ok:
        print("✅ ALL TESTS PASSED!")
        print("You're ready to train Rainbow DQN!")
        print("\nNext step: python scripts/train.py")
    else:
        print("❌ SOME TESTS FAILED")
        print("Please fix the issues above before training.")
        print("\nCommon fixes:")
        print("  - Install missing packages: pip install -r requirements.txt")
        print("  - Install ROMs: AutoROM --accept-license")
    print("="*60)


if __name__ == '__main__':
    main()
