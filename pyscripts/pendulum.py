from gymnasium.wrappers import RecordVideo
import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="rgb_array")
env = RecordVideo(env, video_folder="videos", episode_trigger=lambda x: True)

obs, info = env.reset()
print(f"Starting Observation: {obs}")
print(f"Starting Info: {info}")

for _ in range(500):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()

env.close()