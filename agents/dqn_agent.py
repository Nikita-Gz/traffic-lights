import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import logging

from simulation import TrafficIntersection, SimulationStatesEnum
from agents import TrafficControlAgent
from dqns import QNetwork


logger = logging.getLogger(__name__)


class DQNAgent(TrafficControlAgent):
    def __init__(
        self,
        model: QNetwork,
        target_model: QNetwork | None = None,
        optimizer: optim.Optimizer | None = None,
        memory_size: int = 100000,
    ):
        self.memory_size = memory_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.memory = deque(maxlen=self.memory_size)

        self.model = model.to(self.device)
        self.target_model = (
            target_model.to(self.device) if target_model is not None else None
        )
        self.optimizer = optimizer

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Stores the experience in the memory buffer"""
        if len(self.memory) == self.memory_size:
            logger.warning("Memory buffer is full. Oldest experience will be removed.")
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state: np.ndarray, epsilon: float | None = None) -> int:
        """
        Returns the best action based on the current state,
        unless it decides to explore based on the epsilon value.
        If epsilon is None, it always chooses the best action.
        """
        if epsilon is not None and random.random() < epsilon:
            return random.randrange(TrafficIntersection.action_size)

        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            act_values: torch.Tensor = self.model(state)

        return np.argmax(act_values.cpu().data.numpy())

    def replay(self, batch_size: int, gamma: float = 0.99):
        """
        Trains the model based on the experiences in the memory buffer
        gamma is the discount factor - discounts the future rewards
        """

        if self.optimizer is None or self.target_model is None:
            raise ValueError("The target model and the optimizer must be set to train.")

        if len(self.memory) < batch_size:
            raise ValueError("Not enough experiences in memory to train the model.")

        minibatch: list[tuple[np.ndarray, int, float, np.ndarray, bool]] = (
            random.sample(self.memory, batch_size)
        )
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # The model outputs Q values for all (2) possible actions,
        # but since only the selected action's Q value is needed,
        # .gather() is used to get the Q value of the selected action.
        # Model output shape is (batch_size, action_size),
        # Actions shape after unsqueeze is (batch_size, 1),
        # So, the output shape of gather is (batch_size, 1).
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))

        # Since we assume that we will take the best action in the next state,
        # we call .max(1) to get the max possible Q value for the next state.
        # [0] is called cuz max() returns a tuple of (max_value, max_index).
        # Also, no gradiend is needed for the next Q values, so no_grad() is used.
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0]

        # gamma is the discount factor - discounts the future rewards
        target_q_values = rewards + (gamma * next_q_values * (1 - dones))

        loss = nn.MSELoss()(current_q_values, target_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_model(self):
        """Updates the target model with the weights of the normal model"""
        self.target_model.load_state_dict(self.model.state_dict())

    def get_best_action(
        self, env: TrafficIntersection, simulation_states: list[SimulationStatesEnum]
    ) -> int:
        """Returns the best action based on the current state"""
        state = env.get_state_representation(simulation_states)
        return self.act(state)
