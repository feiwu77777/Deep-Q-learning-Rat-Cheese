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
