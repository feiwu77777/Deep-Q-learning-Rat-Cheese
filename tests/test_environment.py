import numpy as np
import pytest
from src.environment import Environment


def make_env():
    return Environment(grid_size=13, max_time=20, temperature=0.3)


def test_reset_state_shape():
    env = make_env()
    state = env.reset()
    assert state.shape == (2, 5, 5), f"Expected (2, 5, 5), got {state.shape}"


def test_reset_state_dtype():
    env = make_env()
    state = env.reset()
    assert state.dtype == np.float32


def test_act_returns_tuple():
    env = make_env()
    env.reset()
    state, reward, game_over = env.act(0)
    assert state.shape == (2, 5, 5)
    assert isinstance(reward, float)
    assert isinstance(game_over, bool)


def test_edible_cheese_reward():
    env = make_env()
    env.reset()
    # x is always in [3, grid_size-3), so x+1 never hits the wall bounce
    env.board[env.x + 1, env.y] = 0.5
    _, reward, _ = env.act(0)  # action 0: move x+1
    assert reward == pytest.approx(0.5)


def test_poisonous_cheese_reward():
    env = make_env()
    env.reset()
    env.board[env.x + 1, env.y] = -1.0
    _, reward, _ = env.act(0)
    assert reward == pytest.approx(-1.0)


def test_empty_cell_gives_zero_reward():
    env = make_env()
    env.reset()
    env.board[env.x + 1, env.y] = 0.0
    _, reward, _ = env.act(0)
    assert reward == pytest.approx(0.0)


def test_game_over_after_max_time():
    env = make_env()  # max_time=20
    env.reset()
    game_over = False
    for _ in range(25):
        _, _, game_over = env.act(0)
    assert game_over


def test_wall_bounce_x():
    """Rat at x=grid_size-3 (bottom wall) should not exceed grid on action 0."""
    env = make_env()
    env.reset()
    env.x = env.grid_size - 3
    state, _, _ = env.act(0)  # would go out of bounds without bounce
    assert state.shape == (2, 5, 5)
    assert env.x < env.grid_size - 2
