import os
import random

import numpy as np
import torch

from simulation import TrafficIntersection
from rewards import reward_based_on_passed_vehicles
from evaluation import evaluate_agent
from agents import TimeBasedAgent
from plotting import plot_evaluation_stats, plot_car_count_on_directions


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


if __name__ == "__main__":
    set_seed(0)
    env = TrafficIntersection(
        arrival_prob=0.2,
    )
    agent = TimeBasedAgent(light_duration=60)
    result = evaluate_agent(
        env=env,
        agent=agent,
        reward_function=reward_based_on_passed_vehicles,
        num_steps=1000,
    )
    plot_evaluation_stats(result)
    plot_car_count_on_directions(result)
