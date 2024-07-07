from abc import ABC, abstractmethod
from typing import Mapping


class QNetwork(ABC):
    @abstractmethod
    def forward(self, state):
        pass

    @abstractmethod
    def train(self, mode=True):
        pass

    @abstractmethod
    def eval(self):
        pass

    @abstractmethod
    def to(self, device: str) -> "QNetwork":
        pass

    @abstractmethod
    def state_dict(self) -> Mapping:
        pass

    @abstractmethod
    def load_state_dict(self, state_dict: Mapping):
        pass
