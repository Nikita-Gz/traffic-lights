import torch
import torch.nn as nn
from dataclasses import dataclass

from dqns import QNetwork


@dataclass
class SimpleDQNConfig:
    state_size: int
    action_size: int
    hidden_size: int
    extra_hidden_layers: int


class SimpleDQN(nn.Module, QNetwork):
    def __init__(self, config: SimpleDQNConfig):
        super(SimpleDQN, self).__init__()

        self.config = config
        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(config.state_size, config.hidden_size))
        for _ in range(config.extra_hidden_layers):
            self.layers.append(nn.Linear(config.hidden_size, config.hidden_size))
        self.output = nn.Linear(config.hidden_size, config.action_size)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Outputs Q values. Output shape: (batch_size, action_size)"""
        x = state
        for layer in self.layers:
            x = torch.relu(layer(x))
        return self.output(x)
