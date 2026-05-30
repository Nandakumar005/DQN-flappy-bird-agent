# Flappy Bird DQN Agent

This project trains a Deep Q-Network (DQN) agent to play a Flappy Bird-style game. The game environment is built with Pygame, and the reinforcement learning model is built with PyTorch.

The agent learns from four simple state values:

- Bird height
- Bird vertical speed
- Distance to the next pipe
- Height of the next pipe gap

It can choose one of two actions on every frame:

- `0`: do nothing
- `1`: flap/jump

## Project Structure

```text
.
|-- dqn_agent.py       # DQN model, replay buffer, and agent logic
|-- game_env.py        # Flappy Bird environment, physics, rewards, and rendering
|-- main.py            # Training and playback entry point
|-- models/            # Saved trained model files, created during training
|-- sprites_repo/      # Flappy Bird sprite and audio assets
`-- README.md
```

## Requirements

- Python 3.10 or newer
- PyTorch
- Pygame
- NumPy

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows PowerShell:

```bash
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install the required packages:

```bash
pip install torch pygame numpy
```

## How to Run

Train the agent and then watch it play:

```bash
python main.py
```

By default, training runs for `1000` episodes. During training, the best model is saved to:

```text
models/best_model.pth
```

To skip training and only watch the saved model play:

```bash
python main.py --play
```

Make sure `models/best_model.pth` exists before using `--play`. If it does not exist, run training first.

## Controls During Playback

- `R`: reset the game
- `ESC`: quit the game window

## Training Output

While training, the program prints a table with:

- `Ep`: current episode number
- `Reward`: reward earned in the episode
- `Eps`: current exploration rate
- `Avg`: average reward over recent episodes
- `Best`: best reward seen so far
- `Score`: pipes passed in the episode

## Reward System

The environment gives the agent:

- A small positive reward for staying alive
- A larger reward for passing a pipe
- A penalty when it crashes

This encourages the bird to survive longer and pass more pipes.

## Sprite Credits

The visual assets used in this project are from the sprite repository:

[samuelcust/flappy-bird-assets](https://github.com/samuelcust/flappy-bird-assets)

The assets are included in this project under:

```text
sprites_repo/
```

Please check the original repository and its license before reusing or redistributing the assets.

## Notes

- Training quality can vary between runs because reinforcement learning includes randomness.
- If PyTorch detects a CUDA-capable GPU, the agent will use it automatically.
- The `models/` folder is ignored by Git, so trained model files are kept local.
