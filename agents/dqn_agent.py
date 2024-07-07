import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import logging
from dataclasses import dataclass
from typing import Callable

from simulation import TrafficIntersection, SimulationStatesEnum, StepResult
from agents import TrafficControlAgent
from dqns import QNetwork


logger = logging.getLogger(__name__)


@dataclass
class DQNAgentTrainingConfig:
    # Hyperparameters
    learning_rate: float
    gamma: float  # discount factor for future rewards
    initial_epsilon: float  # exploration rate
    epsilon_decay: float  # how much epsilon decreases after each episode
    min_epsilon: float  # minimum value of epsilon
    states_to_consider: list[SimulationStatesEnum]

    # Training loop parameters
    batch_size: int
    update_target_model_every_n_episodes: int
    steps_per_episode: int

    # Other parameters
    memory_size: int


class DQNAgent(TrafficControlAgent):
    def __init__(
        self,
        model_class: type[QNetwork],
        model_config: object,
        training_config: DQNAgentTrainingConfig | None = None,
    ):
        """
        Initializes the DQN agent with thes specified model.
        If training_config is provided, the agent also initializes the optimizer, target model, and memory buffer.
        """
        self.model_class = model_class
        self.model_config = model_config
        self.training_config = training_config

        self.model = model_class(model_config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        if training_config is not None:
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=training_config.learning_rate
            )
            self.target_model = model_class(model_config).to(self.device)
            self.target_model.load_state_dict(self.model.state_dict())
            self.memory = deque(maxlen=training_config.memory_size)
            self.epsilon = training_config.initial_epsilon

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Stores the experience in the memory buffer"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state: np.ndarray, explore: bool = True) -> int:
        """
        Returns the best action based on the current state,
        unless it decides to explore based on the epsilon value.
        Exploration can be disabled with explore=False
        """
        if explore and random.random() < self.epsilon:
            return random.randrange(TrafficIntersection.action_size)

        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            act_values: torch.Tensor = self.model(state)

        return np.argmax(act_values.cpu().data.numpy())

    def replay(self):
        """
        Trains the model based on the experiences in the memory buffer
        gamma is the discount factor - discounts the future rewards
        """

        if self.training_config is None:
            raise ValueError("Training config is not provided.")

        if len(self.memory) < self.training_config.batch_size:
            raise ValueError("Not enough experiences in memory to train the model.")

        minibatch: list[tuple[np.ndarray, int, float, np.ndarray, bool]] = (
            random.sample(self.memory, self.training_config.batch_size)
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
        target_q_values = rewards + (
            self.training_config.gamma * next_q_values * (1 - dones)
        )

        loss = nn.MSELoss()(current_q_values, target_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(
        self,
        env: TrafficIntersection,
        num_episodes: int,
        reward_function: Callable[[StepResult], float],
    ) -> list[float]:
        """
        Trains the model for the specified number of episodes.
        Returns the total rewards for each episode.
        """
        if self.training_config is None:
            raise ValueError("Training config is not provided.")

        total_rewards = []
        for episode_i in range(num_episodes):
            env.reset(
                new_random_seed=episode_i
            )  # new seed ensures different traffic patterns
            current_state = env.get_state_representation(
                self.training_config.states_to_consider
            )
            total_reward_for_episode = 0

            for step_i in range(self.training_config.steps_per_episode):
                action = self.act(current_state, explore=True)
                step_result = env.step(action)

                reward = reward_function(step_result)
                total_reward_for_episode += reward

                next_state = env.get_state_representation(
                    self.training_config.states_to_consider
                )
                done = step_i == self.training_config.steps_per_episode - 1

                self.remember(current_state, action, reward, next_state, done)
                current_state = next_state

                if len(self.memory) >= self.training_config.batch_size:
                    self.replay()

                self.epsilon = max(
                    self.training_config.min_epsilon,
                    self.epsilon * self.training_config.epsilon_decay,
                )

            if (
                episode_i % self.training_config.update_target_model_every_n_episodes
                == 0
            ):
                self.update_target_model()

            logger.info(
                f"Episode {episode_i} - Total reward: {total_reward_for_episode}"
            )
            total_rewards.append(total_reward_for_episode)

        return total_rewards

    def update_target_model(self):
        """Updates the target model with the weights of the normal model"""
        self.target_model.load_state_dict(self.model.state_dict())

    def get_best_action(self, env: TrafficIntersection) -> int:
        """Returns the best action based on the current state"""
        state = env.get_state_representation(self.training_config.states_to_consider)
        return self.act(state, explore=False)
