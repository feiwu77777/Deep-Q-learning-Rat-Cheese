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
        if self.t < len(self.to_draw):
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
