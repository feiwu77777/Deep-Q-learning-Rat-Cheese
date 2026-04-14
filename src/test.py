"""Evaluation entry point — loads a saved model and runs test episodes.

Usage:
    python -m src.test --config config.yaml
"""

import argparse
import os

import yaml

from src.agent import DQN
from src.environment import EnvironmentExploring
from src.model import DQN_CNN


def test(cfg: dict) -> None:
    ec = cfg["env"]
    ac = cfg["agent"]
    tc = cfg["test"]

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
        memory_size=1,   # unused during test
        batch_size=1,    # unused during test
        lr=ac["lr"],
        discount=ac["discount"],
        epsilon=tc["epsilon"],
    )

    model_path = os.path.join(tc["model_dir"], "model.pt")
    agent.load(model_path)

    total_score = 0.0
    for episode in range(tc["epochs"]):
        state = env.reset()
        game_over = False
        win = lose = 0.0

        while not game_over:
            action = agent.act(state)
            state, reward, game_over = env.act(action, train=False)
            if reward > 0:
                win += reward
            elif reward < 0:
                lose -= reward

        env.draw(os.path.join(tc["video_dir"], f"episode{episode}"))
        total_score += win - lose

        print(
            f"Episode {episode + 1}/{tc['epochs']} | "
            f"Win {win:.1f} / Lose {lose:.1f} | "
            f"Avg score {total_score / (episode + 1):.2f}"
        )

    print(f"\nFinal avg score: {total_score / tc['epochs']:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained DQN rat-cheese agent")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    test(cfg)


if __name__ == "__main__":
    main()
