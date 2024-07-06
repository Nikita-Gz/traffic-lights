from typing import Callable
from dataclasses import dataclass, field

from simulation import TrafficIntersection, StepResult
from agents import TrafficControlAgent


@dataclass
class EvaluationStats:
    """Stores stats for each step of the evaluation"""

    rewards: list[float] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    light_states: list[int] = field(default_factory=list)
    vehicles_waiting: list[int] = field(default_factory=list)
    average_wait_times: list[float] = field(default_factory=list)
    max_wait_times: list[int] = field(default_factory=list)

    @property
    def total_reward(self):
        return sum(self.rewards)

    @property
    def average_reward(self):
        return self.total_reward / len(self.rewards)

    @property
    def average_wait_time(self):
        return sum(self.average_wait_times) / len(self.average_wait_times)

    @property
    def max_wait_time(self):
        return max(self.max_wait_times)


def evaluate_agent(
    env: TrafficIntersection,
    agent: TrafficControlAgent,
    reward_function: Callable[[StepResult], float],
    num_steps: int = 1000,
) -> EvaluationStats:
    """
    Records step rewards, actions, and other stats for each step.
    Returns the EvaluationResult object.
    """
    result = EvaluationStats()
    for _ in range(num_steps):
        action = agent.get_best_action(env)
        step_result = env.step(action)
        reward = reward_function(step_result)
        result.rewards.append(reward)
        result.actions.append(action)
        result.light_states.append(env.light_state)
        result.vehicles_waiting.append(
            sum(len(q) for q in env.wait_times_in_directions)
        )

        average_wait_time = (
            env.cumulative_time_waited / env.cumulative_vehicles_passed
            if env.cumulative_vehicles_passed > 0
            else 0
        )
        result.average_wait_times.append(average_wait_time)

        max_wait_times_per_direction = [
            max(wait_times) if wait_times else 0
            for wait_times in step_result.wait_times_in_directions
        ]
        max_wait_time = (
            max(max_wait_times_per_direction) if max_wait_times_per_direction else 0
        )
        result.max_wait_times.append(max_wait_time)
    return result
