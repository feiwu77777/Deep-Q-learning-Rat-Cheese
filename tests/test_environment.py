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


from src.environment import EnvironmentExploring


def make_exploring_env():
    return EnvironmentExploring(grid_size=13, max_time=20, temperature=0.3)


def test_exploring_reset_state_shape():
    env = make_exploring_env()
    state = env.reset()
    assert state.shape == (3, 5, 5), f"Expected (3, 5, 5), got {state.shape}"


def test_exploring_state_dtype():
    env = make_exploring_env()
    state = env.reset()
    assert state.dtype == np.float32


def test_exploring_act_state_shape():
    env = make_exploring_env()
    env.reset()
    state, _, _ = env.act(0, train=True)
    assert state.shape == (3, 5, 5)


def test_exploring_revisit_penalty_in_training():
    """Second visit to same cell penalises the training reward."""
    env = make_exploring_env()
    env.reset()
    env.board[:] = 0.0  # zero board so reward is purely from malus
    # First visit to x+1: malus there is 0 → reward = 0
    _, r1, _ = env.act(0, train=True)
    # Move back: go to x-1 (original x)
    _, _, _ = env.act(1, train=True)
    # Second visit to x+1: malus is now -0.1 → reward = -0.1
    _, r2, _ = env.act(0, train=True)
    assert r2 < r1


def test_exploring_no_penalty_when_not_training():
    """Exploration penalty is not added to reward when train=False."""
    env = make_exploring_env()
    env.reset()
    env.board[:] = 0.0
    # Visit x+1 twice with train=False — both rewards should be 0
    _, r1, _ = env.act(0, train=False)
    _, _, _ = env.act(1, train=False)
    _, r2, _ = env.act(0, train=False)
    assert r1 == pytest.approx(0.0)
    assert r2 == pytest.approx(0.0)
