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
        """Store transition and perform one gradient update if buffer is ready.

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
        game_overs = [b[4] for b in batch]

        q_values = self.model(states)  # (B, 4)

        with torch.no_grad():
            next_q_max = self.model(next_states).max(dim=1).values  # (B,)

        targets = q_values.detach().clone()
        for i in range(self.batch_size):
            if game_overs[i]:
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
