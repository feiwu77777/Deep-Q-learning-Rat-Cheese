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
