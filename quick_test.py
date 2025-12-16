import gymnasium as gym
import ale_py
from AtariWrapper import make_atari_env

# Register ALE environments
gym.register_envs(ale_py)

# Create wrapped environment
print("Creating SpaceInvaders environment...")
env = make_atari_env('ALE/SpaceInvaders-v5')

# Reset and check observation
obs, info = env.reset()
print(f"   Environment created!")
print(f"   Observation shape: {obs.shape}")
print(f"   Observation dtype: {obs.dtype}")
print(f"   Observation range: [{obs.min():.3f}, {obs.max():.3f}]")

print("\nTaking 5 random steps...")
for i in range(5):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"   Step {i+1}: action={action}, reward={reward:.1f}, done={terminated or truncated}")
    if terminated or truncated:
        obs, info = env.reset()
        print("   Episode ended, reset!")

print("Quick test passed! AtariWrapper")
env.close()
