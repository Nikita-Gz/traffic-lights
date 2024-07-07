import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import random
import logging
from typing import Callable

import numpy as np
import torch

from simulation import TrafficIntersection, SimulationStatesEnum, StepResult
from rewards import (
    reward_based_on_wait_time_in_red_queue_only,
    reward_based_on_vehicles_in_red_queue_only,
)
from evaluation import evaluate_agent, EvaluationStats
from agents import TimeBasedAgent, DQNAgent, DQNAgentTrainingConfig
from dqns import SimpleDQN, SimpleDQNConfig
from plotting import (
    plot_evaluation_stats,
    plot_car_count_on_directions,
    plot_rewards_per_episode,
)


logging.basicConfig(level=logging.INFO)


def set_seed(seed=0):
    random.seed(seed)  # Python random module.
    np.random.seed(seed)  # Numpy module.
    torch.manual_seed(seed)  # PyTorch random number generator.
    os.environ["PYTHONHASHSEED"] = str(seed)  # Python hash seed.

    # CUDA related settings
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = (
        True  # Necessary for reproducibility, might slow down training.
    )


def example_with_time_based_agent(
    env: TrafficIntersection, reward_function: Callable[[StepResult], float]
) -> EvaluationStats:
    set_seed(0)
    env.reset()
    agent = TimeBasedAgent(light_duration=30)
    result = evaluate_agent(
        env=env,
        agent=agent,
        reward_function=reward_function,
        num_steps=500,
    )
    plot_evaluation_stats(result)
    plot_car_count_on_directions(result)
    return result


def example_with_dqn_agent(
    env: TrafficIntersection, reward_function: Callable[[StepResult], float]
) -> EvaluationStats:
    set_seed(0)
    env.reset()
    states_to_consider = [
        SimulationStatesEnum.LIGHT_STATE,
        SimulationStatesEnum.LIGHT_IS_TRANSITIONING,
        SimulationStatesEnum.QUEUE_LENGTHS,
        SimulationStatesEnum.AVG_WAIT_TIMES,
    ]
    dqn_model_config = SimpleDQNConfig(
        state_size=env.get_state_representation_length(states_to_consider),
        action_size=2,
        hidden_size=16,
        extra_hidden_layers=2,
    )
    agent_config = DQNAgentTrainingConfig(
        learning_rate=0.001,
        gamma=0.99,
        initial_epsilon=1.0,
        epsilon_decay=0.995,
        min_epsilon=0.05,
        states_to_consider=states_to_consider,
        batch_size=128,
        update_target_model_every_n_episodes=10,
        steps_per_episode=200,
        memory_size=2000,
    )
    agent = DQNAgent(
        model_class=SimpleDQN,
        model_config=dqn_model_config,
        training_config=agent_config,
    )

    # train the agent
    print(f"Training the agent with reward function: {reward_function.__name__}")
    rewards_over_episodes = agent.train(
        env, num_episodes=100, reward_function=reward_function
    )
    result = evaluate_agent(
        env=env,
        agent=agent,
        reward_function=reward_function,
        num_steps=500,
    )
    plot_evaluation_stats(result)
    plot_car_count_on_directions(result)
    plot_rewards_per_episode(rewards_over_episodes)
    return result


if __name__ == "__main__":
    set_seed(0)

    reward_function = reward_based_on_vehicles_in_red_queue_only

    env = TrafficIntersection(arrival_prob=0.3, light_switch_duration=5)
    dqn_evaluation_stats = example_with_dqn_agent(env, reward_function=reward_function)

    time_based_evaluation_stats = example_with_time_based_agent(
        env, reward_function=reward_function
    )

    print("DQN Average wait time:", dqn_evaluation_stats.average_wait_time)
    print(
        "Time-based Average wait time:", time_based_evaluation_stats.average_wait_time
    )
