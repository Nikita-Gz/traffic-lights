from typing import Callable
from dataclasses import dataclass, field

from simulation import TrafficIntersection, StepResult, SimulationStatesEnum
from agents import TrafficControlAgent


@dataclass
class EvaluationStats:
    """Stores stats for each step of the evaluation"""

    rewards: list[float] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    light_states: list[int] = field(default_factory=list)
    passed_vehicles: list[int] = field(default_factory=list)

    # wait_times_per_direction_per_steps[step][direction][vehicle]
    wait_times_per_direction_per_steps: list[list[list[int]]] = field(
        default_factory=list
    )

    def get_wait_times_at_step_at_direction(
        self, step_i: int, direction_i: int
    ) -> list[int]:
        """Returns the wait times for a given step and direction"""
        return self.wait_times_per_direction_per_steps[step_i][direction_i]

    def get_all_wait_times_at_step(self, step_i: int) -> list[int]:
        """Returns all wait times at a given step"""
        return [
            wait_time
            for direction_i in range(4)  # 4 directions
            for wait_time in self.get_wait_times_at_step_at_direction(
                step_i, direction_i
            )
        ]

    @property
    def step_count(self) -> int:
        """Returns the number of steps in the evaluation"""
        return len(self.rewards)

    @property
    def vehicles_waiting(self) -> list[int]:
        """Returns the number of vehicles waiting at each step"""
        return [
            len(self.get_all_wait_times_at_step(step_i))
            for step_i in range(self.step_count)
        ]

    @property
    def wait_times(self) -> list[int]:
        """Returns the total wait time for each step"""
        wait_timse_in_all_steps = []
        for step_i in range(self.step_count):
            wait_timse_in_all_steps.append(sum(self.get_all_wait_times_at_step(step_i)))
        return wait_timse_in_all_steps

    @property
    def average_wait_times(self) -> list[int]:
        """Returns the average wait time for each step"""
        average_wait_times_in_all_steps = []
        for step_i in range(self.step_count):
            wait_times = self.get_all_wait_times_at_step(step_i)
            average_wait_times_in_all_steps.append(
                sum(wait_times) / len(wait_times) if wait_times else 0
            )
        return average_wait_times_in_all_steps

    @property
    def max_wait_times(self) -> list[int]:
        """Returns the max wait time for each step"""
        max_wait_times_in_all_steps = []
        for step_i in range(self.step_count):
            wait_times = self.get_all_wait_times_at_step(step_i)
            max_wait_times_in_all_steps.append(max(wait_times) if wait_times else 0)
        return max_wait_times_in_all_steps

    @property
    def total_reward(self) -> float:
        """Returns the total sum of the rewards for the evaluation"""
        return sum(self.rewards)

    @property
    def average_reward(self) -> float:
        """Returns the average reward for the evaluation"""
        return self.total_reward / len(self.rewards)

    @property
    def average_wait_time(self) -> float:
        """Returns the average wait time for the evaluation"""
        return sum(self.average_wait_times) / len(self.average_wait_times)

    @property
    def max_wait_time(self) -> int:
        """Returns the maximum wait time for the evaluation"""
        return max(self.max_wait_times)


def evaluate_agent(
    env: TrafficIntersection,
    agent: TrafficControlAgent,
    reward_function: Callable[[StepResult], float],
    states_to_consider: list[SimulationStatesEnum],
    num_steps: int = 1000,
) -> EvaluationStats:
    """
    Records step rewards, actions, and other stats for each step.
    Returns the EvaluationResult object.
    """
    result = EvaluationStats()
    for _ in range(num_steps):
        action = agent.get_best_action(env, states_to_consider)
        step_result = env.step(action)
        reward = reward_function(step_result)
        result.rewards.append(reward)
        result.actions.append(action)
        result.light_states.append(env.light_state)
        result.passed_vehicles.append(step_result.passed_vehicles)
        result.wait_times_per_direction_per_steps.append(env.wait_times_per_direction)
    return result
