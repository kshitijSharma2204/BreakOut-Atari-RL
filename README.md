# Breakout DQN

A Deep Q-Network (DQN) agent trained to play Atari Breakout using Stable-Baselines3.

## Demo

<p align="left">
  <img src="output/rl-video-episode-7.gif" width="320" alt="Breakout Gameplay" />
</p>

You can also view the full-resolution video at `output/rl-video-episode-7.mp4`.

## Installation

1. **Clone the repo**  
   ```bash
   git clone git@github.com:kshitijSharma2204/BreakOut-Atari-RL.git
   cd BreakOut-Atari-RL
   ```

2. **Create a virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
## Usage

### 1. Training

Train the DQN agent:
```bash
python scripts/train_dqn_breakout.py
This will save the final model to breakout_dqn_final.zip.
```

### 2. Playback & Recording
Generate gameplay videos (no on-screen display):
```bash
python scripts/play_dqn_breakout.py
```
Each episode will be recorded as output/rl-video-episode-<n>.mp4.
