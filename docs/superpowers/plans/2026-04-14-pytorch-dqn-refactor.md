# PyTorch DQN Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite the Keras rat-cheese DQN pipeline as a clean PyTorch `src/` package with config-driven training and testing entry points.

**Architecture:** `model.py` defines a pure `nn.Module`; `environment.py` provides `Environment` (base, 2-channel state) and `EnvironmentExploring` (subclass, 3-channel); `agent.py` owns the replay buffer and learning logic; `train.py`/`test.py` are CLI entry points that read `config.yaml`.

**Tech Stack:** Python 3, PyTorch, NumPy, OpenCV (`cv2`), scikit-video (`skvideo`), PyYAML, pytest

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `src/__init__.py` | Create | Package marker |
| `src/model.py` | Create | `DQN_CNN(nn.Module)` — forward pass only |
| `src/environment.py` | Create | `Environment` + `EnvironmentExploring` |
| `src/agent.py` | Create | `DQN` — replay buffer, epsilon-greedy, reinforce, save/load |
| `src/train.py` | Create | Training entry point |
| `src/test.py` | Create | Test/evaluation entry point |
| `config.yaml` | Create | All hyperparameters |
| `requirements.txt` | Create | Python dependencies |
| `tests/__init__.py` | Create | Test package marker |
| `tests/test_model.py` | Create | Model forward pass shape tests |
| `tests/test_environment.py` | Create | Environment reset/act/reward tests |
| `tests/test_agent.py` | Create | Agent act/reinforce/save/load tests |
| `main.ipynb` | Modify | Replace first cell with deprecation notice |

---

### Task 1: Project Scaffold

**Files:**
- Create: `src/__init__.py`
- Create: `tests/__init__.py`
- Create: `requirements.txt`
- Create: `config.yaml`

- [ ] **Step 1: Create directories and empty package markers**

```bash
mkdir -p src tests model video/train video/test
touch src/__init__.py tests/__init__.py
```

- [ ] **Step 2: Create `requirements.txt`**

```
torch>=2.0
numpy
opencv-python
scikit-video
pyyaml
pytest
```

- [ ] **Step 3: Create `config.yaml`**

```yaml
env:
  grid_size: 13
  max_time: 200
  temperature: 0.3

agent:
  epsilon_start: 1.0
  epsilon_end: 0.01
  decay_rate: 0.001
  memory_size: 2000
  batch_size: 32
  lr: 0.1
  discount: 0.99
  n_state: 3  # 2 for base Environment, 3 for EnvironmentExploring

train:
  epochs: 50
  video_every: 10
  model_dir: model/
  video_dir: video/train/

test:
  epochs: 5
  epsilon: 0.1
  model_dir: model/
  video_dir: video/test/
```

- [ ] **Step 4: Commit scaffold**

```bash
git add src/__init__.py tests/__init__.py requirements.txt config.yaml
git commit -m "feat: add project scaffold for PyTorch DQN rewrite"
```

---

### Task 2: `src/model.py` — DQN_CNN

**Files:**
- Create: `src/model.py`
- Create: `tests/test_model.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_model.py`:

```python
import torch
from src.model import DQN_CNN


def test_forward_2channel():
    model = DQN_CNN(n_state=2)
    x = torch.zeros(4, 2, 5, 5)
    out = model(x)
    assert out.shape == (4, 4), f"Expected (4, 4), got {out.shape}"


def test_forward_3channel():
    model = DQN_CNN(n_state=3)
    x = torch.zeros(8, 3, 5, 5)
    out = model(x)
    assert out.shape == (8, 4), f"Expected (8, 4), got {out.shape}"


def test_output_not_all_zeros_on_random_input():
    model = DQN_CNN(n_state=3)
    x = torch.randn(2, 3, 5, 5)
    out = model(x)
    assert not torch.all(out == 0)
```

- [ ] **Step 2: Run tests — expect failure**

```bash
pytest tests/test_model.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.model'`

- [ ] **Step 3: Implement `src/model.py`**

```python
import torch.nn as nn


class DQN_CNN(nn.Module):
    """CNN Q-network mapping a (n_state, 5, 5) state window to 4 Q-values.

    Input shape:  (batch, n_state, 5, 5)  — channels-first (PyTorch convention)
    Output shape: (batch, 4)              — one Q-value per action

    Spatial reduction: 5x5 -> Conv2d(k=2) -> 4x4 -> Conv2d(k=2) -> 3x3
    Flatten: 64 * 3 * 3 = 576
    """

    def __init__(self, n_state: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(n_state, 32, kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(576, 4),
        )

    def forward(self, x):
        return self.net(x)
```

- [ ] **Step 4: Run tests — expect pass**

```bash
pytest tests/test_model.py -v
```

Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add src/model.py tests/test_model.py
git commit -m "feat: add DQN_CNN nn.Module with shape tests"
```

---

### Task 3: `src/environment.py` — Base Environment

**Files:**
- Create: `src/environment.py` (base class only)
- Create: `tests/test_environment.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_environment.py`:

```python
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
```

- [ ] **Step 2: Run tests — expect failure**

```bash
pytest tests/test_environment.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.environment'`

- [ ] **Step 3: Implement `Environment` in `src/environment.py`**

```python
import os

import cv2
import numpy as np
import skvideo.io


class Environment:
    """Rat-and-cheese grid environment (2-channel state: board + position).

    Grid layout: (grid_size+4) x (grid_size+4) with 2-cell walls on all sides.
    The rat sees only a 5x5 window centred on itself.

    State channels (axis 0):
        0 — board view: +0.5 edible, -1.0 poisonous, 0 empty
        1 — position view: -1 walls, 0 empty, +1 rat

    Actions: 0=down(x+1), 1=up(x-1), 2=right(y+1), 3=left(y-1)
    Reward: board value at landing cell (+0.5 or -1.0 or 0.0)
    """

    def __init__(self, grid_size: int = 13, max_time: int = 200, temperature: float = 0.3):
        self.grid_size = grid_size + 4  # 2-cell border on each side
        self.max_time = max_time
        self.temperature = temperature
        self.scale = 16
        self._alloc_frames()

    def _alloc_frames(self):
        h = self.grid_size * self.scale
        self.to_draw = np.zeros((self.max_time + 2, h, h, 3), dtype=np.uint8)

    def reset(self):
        """Reset grid, scatter cheese, place rat. Returns state (2, 5, 5)."""
        self.x = int(np.random.randint(3, self.grid_size - 3))
        self.y = int(np.random.randint(3, self.grid_size - 3))
        self.t = 0
        self._alloc_frames()

        bonus = 0.5 * np.random.binomial(1, self.temperature,
                                          size=(self.grid_size, self.grid_size)).astype(np.float32)
        malus = -1.0 * np.random.binomial(1, self.temperature,
                                           size=(self.grid_size, self.grid_size)).astype(np.float32)
        malus[bonus > 0] = 0.0
        self.board = (bonus + malus).astype(np.float32)
        self.board[self.x, self.y] = 0.0  # clear rat's starting cell

        return self._make_state()

    def _make_state(self):
        board_view = self.board[self.x - 2:self.x + 3, self.y - 2:self.y + 3].copy()
        pos_view = self._make_position()[self.x - 2:self.x + 3, self.y - 2:self.y + 3]
        return np.stack([board_view, pos_view], axis=0).astype(np.float32)

    def _make_position(self):
        pos = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        pos[0:2, :] = -1.0
        pos[:, 0:2] = -1.0
        pos[-2:, :] = -1.0
        pos[:, -2:] = -1.0
        pos[self.x, self.y] = 1.0
        return pos

    def act(self, action: int, train: bool = False):
        """Move rat, collect cheese, advance time. Returns (state, reward, game_over)."""
        self.get_frame(self.t)

        if action == 0:
            self.x = self.x - 1 if self.x == self.grid_size - 3 else self.x + 1
        elif action == 1:
            self.x = self.x + 1 if self.x == 2 else self.x - 1
        elif action == 2:
            self.y = self.y - 1 if self.y == self.grid_size - 3 else self.y + 1
        elif action == 3:
            self.y = self.y + 1 if self.y == 2 else self.y - 1

        reward = float(self.board[self.x, self.y])
        self.board[self.x, self.y] = 0.0
        self.t += 1
        game_over = bool(self.t > self.max_time)

        return self._make_state(), reward, game_over

    def get_frame(self, t: int):
        """Render the current board state into the frame buffer at index t."""
        b = np.full((self.grid_size, self.grid_size, 3), 128, dtype=np.float32)
        b[self.board > 0, 0] = 255   # red channel — edible cheese
        b[self.board < 0, 2] = 255   # blue channel — poisonous cheese
        b[self.x, self.y, :] = 255   # white — rat
        b[:2, :, :] = 60             # dark walls
        b[-2:, :, :] = 60
        b[:, :2, :] = 60
        b[:, -2:, :] = 60
        b = cv2.resize(b, None, fx=self.scale, fy=self.scale,
                       interpolation=cv2.INTER_NEAREST)
        self.to_draw[t] = np.clip(b, 0, 255).astype(np.uint8)

    def draw(self, path: str):
        """Write recorded frames to an .mp4 file at `path + '.mp4'`."""
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        skvideo.io.vwrite(path + ".mp4", self.to_draw[:self.t])
```

- [ ] **Step 4: Run tests — expect pass**

```bash
pytest tests/test_environment.py -v
```

Expected: `8 passed`

- [ ] **Step 5: Commit**

```bash
git add src/environment.py tests/test_environment.py
git commit -m "feat: add base Environment with 2-channel state"
```

---

### Task 4: `EnvironmentExploring` Subclass

**Files:**
- Modify: `src/environment.py` (append subclass)
- Modify: `tests/test_environment.py` (append tests)

- [ ] **Step 1: Write failing tests**

Append to `tests/test_environment.py`:

```python
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
    _, _ , _ = env.act(1, train=False)
    _, r2, _ = env.act(0, train=False)
    assert r1 == pytest.approx(0.0)
    assert r2 == pytest.approx(0.0)
```

- [ ] **Step 2: Run new tests — expect failure**

```bash
pytest tests/test_environment.py::test_exploring_reset_state_shape -v
```

Expected: `ImportError` (EnvironmentExploring not yet defined)

- [ ] **Step 3: Append `EnvironmentExploring` to `src/environment.py`**

```python
class EnvironmentExploring(Environment):
    """Environment that penalises revisiting cells to encourage exploration.

    Adds a malus_position grid that decrements by 0.1 each time the rat
    lands on a cell. During training this penalty is added to the reward.

    State channels (axis 0):
        0 — malus view: starts at 0, decreases -0.1 per visit
        1 — board view: +0.5 edible, -1.0 poisonous, 0 empty
        2 — position view: -1 walls, 0 empty, +1 rat
    """

    def reset(self):
        super().reset()  # sets self.x, self.y, self.board, self.t
        self.malus_position = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.malus_position[self.x, self.y] = -0.1
        return self._make_exploring_state()

    def _make_exploring_state(self):
        malus_view = self.malus_position[self.x - 2:self.x + 3, self.y - 2:self.y + 3].copy()
        board_view = self.board[self.x - 2:self.x + 3, self.y - 2:self.y + 3].copy()
        pos_view = self._make_position()[self.x - 2:self.x + 3, self.y - 2:self.y + 3]
        return np.stack([malus_view, board_view, pos_view], axis=0).astype(np.float32)

    def act(self, action: int, train: bool = False):
        self.get_frame(self.t)

        if action == 0:
            self.x = self.x - 1 if self.x == self.grid_size - 3 else self.x + 1
        elif action == 1:
            self.x = self.x + 1 if self.x == 2 else self.x - 1
        elif action == 2:
            self.y = self.y - 1 if self.y == self.grid_size - 3 else self.y + 1
        elif action == 3:
            self.y = self.y + 1 if self.y == 2 else self.y - 1

        cheese_reward = float(self.board[self.x, self.y])
        self.board[self.x, self.y] = 0.0

        if train:
            reward = cheese_reward + float(self.malus_position[self.x, self.y])
        else:
            reward = cheese_reward

        self.malus_position[self.x, self.y] -= 0.1
        self.t += 1
        game_over = bool(self.t > self.max_time)

        return self._make_exploring_state(), reward, game_over
```

- [ ] **Step 4: Run all environment tests — expect pass**

```bash
pytest tests/test_environment.py -v
```

Expected: `13 passed`

- [ ] **Step 5: Commit**

```bash
git add src/environment.py tests/test_environment.py
git commit -m "feat: add EnvironmentExploring with revisit penalty and 3-channel state"
```

---

### Task 5: `src/agent.py` — DQN

**Files:**
- Create: `src/agent.py`
- Create: `tests/test_agent.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_agent.py`:

```python
import os

import numpy as np
import pytest
import torch

from src.agent import DQN
from src.model import DQN_CNN


def make_agent(n_state=3, memory_size=100, batch_size=4):
    model = DQN_CNN(n_state=n_state)
    return DQN(model=model, n_state=n_state, memory_size=memory_size,
               batch_size=batch_size, lr=0.1, discount=0.99, epsilon=0.5)


def random_state(n_state=3):
    return np.random.rand(n_state, 5, 5).astype(np.float32)


def test_act_returns_valid_action():
    agent = make_agent()
    action = agent.act(random_state())
    assert action in [0, 1, 2, 3]


def test_act_greedy_when_epsilon_zero():
    agent = make_agent()
    agent.epsilon = 0.0
    state = random_state()
    actions = [agent.act(state) for _ in range(20)]
    assert len(set(actions)) == 1, "epsilon=0 should always return same action"


def test_reinforce_returns_zero_when_buffer_insufficient():
    agent = make_agent(batch_size=8)
    loss = agent.reinforce(random_state(), random_state(), 0, 0.5, False)
    assert loss == 0.0


def test_reinforce_returns_positive_loss_after_buffer_filled():
    agent = make_agent(batch_size=4)
    loss = 0.0
    for _ in range(5):
        loss = agent.reinforce(random_state(), random_state(),
                               int(np.random.randint(0, 4)), float(np.random.randn()), False)
    assert isinstance(loss, float)
    assert loss >= 0.0


def test_save_and_load_weights_match(tmp_path):
    agent = make_agent()
    path = str(tmp_path / "model.pt")
    agent.save(path)
    assert os.path.exists(path)

    agent2 = make_agent()
    agent2.load(path)

    state = torch.FloatTensor(random_state()).unsqueeze(0)
    with torch.no_grad():
        out1 = agent.model(state)
        out2 = agent2.model(state)
    assert torch.allclose(out1, out2), "Loaded model should produce identical output"
```

- [ ] **Step 2: Run tests — expect failure**

```bash
pytest tests/test_agent.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.agent'`

- [ ] **Step 3: Implement `src/agent.py`**

```python
from collections import deque

import numpy as np
import torch
import torch.nn as nn

from src.model import DQN_CNN


class DQN:
    """Deep Q-Network agent with epsilon-greedy exploration and experience replay.

    Args:
        model:       DQN_CNN instance (the Q-network)
        n_state:     number of state channels (2 for Environment, 3 for EnvironmentExploring)
        memory_size: maximum replay buffer size
        batch_size:  number of transitions sampled per gradient step
        lr:          SGD learning rate
        discount:    Bellman discount factor γ
        epsilon:     initial exploration probability
    """

    def __init__(
        self,
        model: DQN_CNN,
        n_state: int = 3,
        memory_size: int = 2000,
        batch_size: int = 32,
        lr: float = 0.1,
        discount: float = 0.99,
        epsilon: float = 1.0,
    ):
        self.model = model
        self.n_state = n_state
        self.batch_size = batch_size
        self.discount = discount
        self.epsilon = epsilon
        self.n_action = 4

        self.memory: deque = deque(maxlen=memory_size)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def act(self, state: np.ndarray) -> int:
        """Epsilon-greedy action: random with prob epsilon, else argmax Q."""
        if np.random.rand() < self.epsilon:
            return int(np.random.randint(0, self.n_action))
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0)  # (1, n_state, 5, 5)
            return int(self.model(s).argmax(dim=1).item())

    def reinforce(
        self,
        s: np.ndarray,
        ns: np.ndarray,
        action: int,
        reward: float,
        game_over: bool,
    ) -> float:
        """Store transition; if buffer ready, sample a batch and update weights.

        Returns the MSE loss for this update (0.0 if buffer not yet full enough).
        """
        self.memory.append((s, ns, action, reward, game_over))
        if len(self.memory) < self.batch_size:
            return 0.0

        indices = np.random.choice(len(self.memory), self.batch_size, replace=False)
        batch = [self.memory[i] for i in indices]

        states = torch.FloatTensor(np.array([b[0] for b in batch]))       # (B, n_state, 5, 5)
        next_states = torch.FloatTensor(np.array([b[1] for b in batch]))  # (B, n_state, 5, 5)
        actions = [b[2] for b in batch]
        rewards = [b[3] for b in batch]
        dones = [b[4] for b in batch]

        q_values = self.model(states)  # (B, 4)

        with torch.no_grad():
            next_q_max = self.model(next_states).max(dim=1).values  # (B,)

        targets = q_values.detach().clone()
        for i in range(self.batch_size):
            if dones[i]:
                targets[i, actions[i]] = rewards[i]
            else:
                targets[i, actions[i]] = rewards[i] + self.discount * next_q_max[i].item()

        targets = torch.clamp(targets, -3.0, 3.0)

        loss = self.criterion(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(loss.item())

    def save(self, path: str = "model/model.pt") -> None:
        torch.save(self.model.state_dict(), path)

    def load(self, path: str = "model/model.pt") -> None:
        self.model.load_state_dict(torch.load(path, map_location="cpu"))
        self.model.eval()
```

- [ ] **Step 4: Run agent tests — expect pass**

```bash
pytest tests/test_agent.py -v
```

Expected: `5 passed`

- [ ] **Step 5: Run full test suite**

```bash
pytest tests/ -v
```

Expected: all tests pass across model, environment, and agent.

- [ ] **Step 6: Commit**

```bash
git add src/agent.py tests/test_agent.py
git commit -m "feat: add DQN agent with experience replay and PyTorch training step"
```

---

### Task 6: `src/train.py` — Training Entry Point

**Files:**
- Create: `src/train.py`

- [ ] **Step 1: Create `src/train.py`**

```python
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
```

- [ ] **Step 2: Verify import is clean**

```bash
python -c "from src.train import train; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/train.py
git commit -m "feat: add train.py entry point with epsilon decay and video export"
```

---

### Task 7: `src/test.py` — Evaluation Entry Point

**Files:**
- Create: `src/test.py`

- [ ] **Step 1: Create `src/test.py`**

```python
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
```

- [ ] **Step 2: Verify import is clean**

```bash
python -c "from src.test import test; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/test.py
git commit -m "feat: add test.py evaluation entry point with mp4 export"
```

---

### Task 8: Deprecate `main.ipynb`

**Files:**
- Modify: `main.ipynb` (first cell only)

- [ ] **Step 1: Replace first cell source in `main.ipynb`**

The notebook's first cell is a markdown cell with `### Summary`. Replace its entire source with:

```markdown
> **DEPRECATED** — This notebook uses Keras/TensorFlow and is no longer maintained.
>
> The PyTorch rewrite lives in `src/`. To train:
> ```bash
> python -m src.train --config config.yaml
> ```
> To evaluate a saved model:
> ```bash
> python -m src.test --config config.yaml
> ```
> All hyperparameters are in `config.yaml`.
> Architecture details: `docs/superpowers/specs/2026-04-14-pytorch-dqn-refactor-design.md`

---

### Archive — original Keras notebook outline
0. Imports
1. Deep Q Neural Network (Keras/TF)
2. Environment
3. Training
4. Testing
```

- [ ] **Step 2: Run full test suite one final time**

```bash
pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 3: Commit**

```bash
git add main.ipynb
git commit -m "chore: deprecate main.ipynb in favour of src/ PyTorch rewrite"
```
