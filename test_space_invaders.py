import gymnasium as gym
import ale_py 

# Register ALE environments with Gymnasium
gym.register_envs(ale_py)

env = gym.make("ALE/SpaceInvaders-v5", render_mode=None)

obs, info = env.reset()
print("Observation shape:", obs.shape)
print("Action space:", env.action_space)

env.close()
