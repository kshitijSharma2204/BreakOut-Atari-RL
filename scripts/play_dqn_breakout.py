# play_dqn_breakout_record_only_highres.py

import time
import numpy as np
import torch
import gym
from stable_baselines3 import DQN
from gym.wrappers import RecordVideo, AtariPreprocessing, FrameStack

# — CONFIGURATION —
ENV_ID       = "ALE/Breakout-v5"
MODEL_PATH   = "breakout_dqn_final.zip"
VIDEO_FOLDER = "videos"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

def make_env():
    # 1) Raw env in rgb_array mode for high-res frames (≈210×160)
    raw_env = gym.make(ENV_ID, render_mode="rgb_array", frameskip=1)
    # 2) Wrap it in RecordVideo so we capture full-size frames
    rec_env = RecordVideo(
        raw_env,
        video_folder=VIDEO_FOLDER,
        episode_trigger=lambda _: True
    )
    # 3) Now apply exactly the same preprocessing used during training
    proc_env = AtariPreprocessing(
        rec_env,
        grayscale_obs=True,
        frame_skip=4,
        scale_obs=False
    )
    # 4) Stack the last 4 frames for our DQN
    env = FrameStack(proc_env, num_stack=4)
    return env

if __name__ == "__main__":
    # Load the trained DQN agent
    model = DQN.load(MODEL_PATH, device=DEVICE)

    # Build our recording-only environment
    env = make_env()

    # Initial reset (Gym ≥0.26 returns obs, info)
    reset_out = env.reset()
    obs = reset_out[0] if isinstance(reset_out, (tuple, list)) else reset_out
    obs = np.array(obs)

    try:
        while True:
            # Predict action
            action, _ = model.predict(obs, deterministic=True)

            # Step the env (captures full-size frame to video folder)
            step_out = env.step(action)
            if len(step_out) == 5:
                obs, reward, terminated, truncated, info = step_out
                done = terminated or truncated
            else:
                obs, reward, done, info = step_out

            # Convert LazyFrames → ndarray
            obs = np.array(obs)

            # When episode ends, reset to start a new video file
            if done:
                reset_out = env.reset()
                obs = reset_out[0] if isinstance(reset_out, (tuple, list)) else reset_out
                obs = np.array(obs)

            # Throttle to ~30 FPS
            time.sleep(1 / 30)

    except KeyboardInterrupt:
        pass
    finally:
        env.close()
