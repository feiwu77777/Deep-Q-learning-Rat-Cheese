"""Training entry point.

Usage:
    python -m src.train --config config.yaml
"""

import argparse
import os

import numpy as np
import yaml

from src.agent import DQN
from src.environment import EnvironmentExploring
from src.model import DQN_CNN


def init_memory(agent: DQN, env: EnvironmentExploring) -> None:
    """Pre-fill the replay buffer with random transitions before training."""
    state = env.reset()
    for _ in range(agent.batch_size):
        action = int(np.random.randint(0, 4))
        next_state, reward, game_over = env.act(action, train=True)
        agent.memory.append((state, next_state, action, reward, game_over))
        state = env.reset() if game_over else next_state


def train(cfg: dict) -> None:
    ec = cfg["env"]
    ac = cfg["agent"]
    tc = cfg["train"]

    assert ac["n_state"] == 3, (
        f"train.py uses EnvironmentExploring which requires n_state=3, got {ac['n_state']}"
    )

    os.makedirs(tc["model_dir"], exist_ok=True)
    os.makedirs(tc["video_dir"], exist_ok=True)

    env = EnvironmentExploring(
        grid_size=ec["grid_size"],
        max_time=ec["max_time"],
        temperature=ec["temperature"],
    )
    model = DQN_CNN(n_state=ac["n_state"])
    agent = DQN(
        model=model,
        n_state=ac["n_state"],
        memory_size=ac["memory_size"],
        batch_size=ac["batch_size"],
        lr=ac["lr"],
        discount=ac["discount"],
        epsilon=ac["epsilon_start"],
    )

    init_memory(agent, env)

    decay_step = 0
    for epoch in range(tc["epochs"]):
        state = env.reset()
        game_over = False
        win = lose = loss = 0.0

        while not game_over:
            decay_step += 1
            agent.epsilon = ac["epsilon_end"] + (
                ac["epsilon_start"] - ac["epsilon_end"]
            ) * np.exp(-ac["decay_rate"] * decay_step)

            action = agent.act(state)
            next_state, reward, game_over = env.act(action, train=True)
            loss = agent.reinforce(state, next_state, action, reward, game_over)

            if reward > 0:
                win += reward
            elif reward < 0:
                lose -= reward

            state = next_state

        if epoch % tc["video_every"] == 0:
            env.draw(os.path.join(tc["video_dir"], f"epoch{epoch}"))

        agent.save(os.path.join(tc["model_dir"], "model.pt"))

        print(
            f"Epoch {epoch:03d}/{tc['epochs']:03d} | "
            f"Loss {loss:.4f} | "
            f"Win/Lose {win:.1f}/{lose:.1f} ({win - lose:.1f})"
        )


def main():
    parser = argparse.ArgumentParser(description="Train DQN rat-cheese agent")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train(cfg)


if __name__ == "__main__":
    main()
