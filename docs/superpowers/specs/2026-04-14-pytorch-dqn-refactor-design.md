# PyTorch DQN Refactor Design

**Date:** 2026-04-14
**Status:** Approved

## Goal

Deprecate `main.ipynb` (Keras/TensorFlow) and rewrite the full Deep Q-Learning training pipeline in PyTorch, split across a clean `src/` Python package.

---

## File Structure

```
Deep-Q-learning-Rat-Cheese/
├── src/
│   ├── __init__.py
│   ├── model.py        # DQN_CNN as nn.Module
│   ├── environment.py  # Environment (base) + EnvironmentExploring (subclass)
│   ├── agent.py        # DQN: replay buffer, epsilon-greedy, reinforce, save/load
│   ├── train.py        # training entry point
│   └── test.py         # test entry point
├── config.yaml         # all hyperparameters
├── model/              # saved weights (.pt)
├── video/
│   ├── train/
│   └── test/
└── main.ipynb          # deprecation notice + pointer to src/
```

---

## Section 1: model.py — `DQN_CNN(nn.Module)`

- **Input:** `(batch, n_state, 5, 5)` — PyTorch channels-first (vs Keras channels-last)
- **Architecture:** `Conv2d(n_state→32, kernel=2, ReLU)` → `Conv2d(32→64, kernel=2, ReLU)` → `Flatten` → `Linear(→4)`
- **Responsibility:** pure forward pass only — no optimizer, no training logic
- Weights saved as `.pt` via `state_dict`

---

## Section 2: agent.py — `DQN`

- **Replay buffer:** `collections.deque(maxlen=memory_size)`
- **`act(state)`:** epsilon-greedy — random action with prob `epsilon`, else `argmax` of Q-values
- **`reinforce(s, next_s, action, reward, game_over)`:**
  - Appends transition to replay buffer
  - Samples a random batch
  - Computes Bellman targets with `torch.no_grad()`
  - Clips targets to `[-3, 3]`
  - Runs one SGD step with `nn.MSELoss`
- **`save(path)` / `load(path)`:** `torch.save` / `torch.load` on `model.state_dict()`
- **Bug fix from original:** corrects `predcit` typo; target computation properly detached from gradient graph
- **Optimizer:** SGD (lr + momentum from config), owned by agent
- CPU-only — no CUDA support

---

## Section 3: environment.py

### `Environment` (base, 2-channel state)

- Grid size: `(grid_size + 4) × (grid_size + 4)` with 2-cell walls on all sides
- **`reset()`:** places rat at random valid position, scatters cheese via Binomial draw with `temperature`, returns initial 5×5 state window `[board, position]`
- **`act(action, train=False)`:** moves rat (bounces off walls), updates cheese board, increments timer. Reward: `+0.5` (edible), `−1.0` (poisonous). Returns `(state, reward, game_over)`
- **`get_frame(t)`:** renders board to `self.to_draw[t]` using OpenCV resize (scale=16)
- **`draw(path)`:** writes frames to `.mp4` via `skvideo.io.vwrite`
- State channels: `[board, position]`

### `EnvironmentExploring(Environment)` (3-channel state)

- Adds `malus_position` grid (same shape as board), initialized to zeros on `reset()`
- On each step: `malus_position[rat_x, rat_y] -= 0.1` (discourages revisiting cells)
- Training reward: `board_reward + malus_position[rat_x, rat_y]`
- State channels: `[malus_position, board, position]`
- `n_state=3` in config when using this environment

---

## Section 4: config.yaml

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
  n_state: 3  # 2 for base env, 3 for exploring

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

---

## Section 5: train.py & test.py

**`train.py`** — CLI entry point: `python -m src.train --config config.yaml`
- Loads config via PyYAML
- Instantiates `EnvironmentExploring` + `DQN`
- Pre-fills replay buffer with random transitions (`init_memory`)
- Runs training loop with exponential epsilon decay: `ε = ε_end + (ε_start - ε_end) * exp(-decay_rate * step)`
- Saves model checkpoint each epoch to `model_dir`
- Saves `.mp4` every `video_every` epochs to `video_dir`
- Prints per-epoch: loss, win count, lose count, net score

**`test.py`** — CLI entry point: `python -m src.test --config config.yaml`
- Loads config + saved model weights
- Runs `test.epochs` episodes with `train=False` (no exploration penalty in reward)
- Prints per-episode: win/lose counts, running average score
- Saves each episode as `.mp4` to `video_dir`

---

## Deprecation

`main.ipynb` will have its first cell replaced with a deprecation notice:
- Marks it as deprecated (Keras/TensorFlow)
- Points to `src/` and `config.yaml` for the PyTorch rewrite
- All other cells left intact for reference

---

## Key Improvements Over Original

| Issue | Fix |
|---|---|
| `self.model.predcit(...)` typo | Corrected to `predict` / proper PyTorch forward call |
| `EnvironmentExploring` undefined | Properly implemented as subclass |
| Keras/TF dependency | Replaced with PyTorch `nn.Module` |
| Hardcoded hyperparameters | Moved to `config.yaml` |
| No entry points | `train.py` and `test.py` as `python -m src.train` |
| Weights in `.h5`/`.json` | Replaced with `.pt` (`state_dict`) |
