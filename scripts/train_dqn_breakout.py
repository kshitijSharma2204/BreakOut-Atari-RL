import os
import torch
import gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from gym.wrappers import AtariPreprocessing, FrameStack

# — CONFIGURATION —
ENV_ID = "ALE/Breakout-v5"
TOTAL_TIMESTEPS = 1_000_000  # ↑ bump this up (e.g. 2_000_000+) to get stronger performance
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_PATH = "breakout_dqn_final"  # will save to breakout_dqn_final.zip

def make_env():
    env = gym.make(ENV_ID, frameskip=1)
    env = AtariPreprocessing(env, grayscale_obs=True, frame_skip=4, scale_obs=False)
    env = FrameStack(env, num_stack=4)
    return env

if __name__ == "__main__":
    os.makedirs(os.path.dirname(SAVE_PATH) or ".", exist_ok=True)

    # Vectorized training environment
    train_env = DummyVecEnv([make_env])

    # Instantiate and train
    model = DQN(
        policy="CnnPolicy",
        env=train_env,
        verbose=1,
        buffer_size=100_000,
        learning_rate=1e-4,
        device=DEVICE,
    )
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    model.save(SAVE_PATH)

    print(f"Training complete — model saved to {SAVE_PATH}.zip")
