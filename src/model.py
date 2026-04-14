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
