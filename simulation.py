from dataclasses import dataclass, field
import random
import logging
from collections import deque
from typing import Literal


@dataclass
class StepResult:
    """Stores the results of a step in the environment, not cumulative stats"""

    step_number: int
    new_vehicles: int
    passed_vehicles: int
    waited_time: int
    wait_times_in_directions: list[list[int]] = field(default_factory=list)


class TrafficIntersection:
    def __init__(
        self,
        arrival_prob: float = 0.1,
        random_seed: int = 0,
        logger: logging.Logger | None = None,
    ):
        self.arrival_prob = arrival_prob
        self.random_seed = random_seed
        self.logger = logger

        # each item in the queue represents a vehicle and the value represents the time it has waited
        self.wait_times_in_directions: list[deque[int]] = [
            deque() for _ in range(4)
        ]  # N, S, E, W

        self.light_state = 0  # 0: N-S Green, 1: E-W Green

        self.step_count = 0
        self.cumulative_time_waited = 0
        self.cumulative_vehicles_passed = 0

        self.random_generator = random.Random(self.random_seed)

    def _log(self, level: int, message: str):
        if self.logger is not None:
            self.logger.log(level, message)

    def step(self, action: Literal[0, 1]) -> StepResult:
        """Take a step in the environment, update the state and return the reward"""

        self._log(logging.DEBUG, f"Step {self.step_count}, action: {action}")

        # Update the state
        self._process_traffic_light(action)
        passed_this_step = self._process_cars_passing()
        new_vehicles_this_step = self._add_new_vehicles()
        waited_this_step = self._update_waiting_times()
        step_result = StepResult(
            step_number=self.step_count,
            new_vehicles=new_vehicles_this_step,
            passed_vehicles=passed_this_step,
            waited_time=waited_this_step,
            wait_times_in_directions=self.wait_times_per_each_car_per_direction,
        )

        self._log(logging.DEBUG, f"Step result: {step_result}")

        self.step_count += 1

        return step_result

    def _process_traffic_light(self, action: Literal[0, 1]):
        """If action is 1, switch the traffic light"""
        if action == 1:
            self.light_state = 1 - self.light_state

    def _process_cars_passing(self) -> int:
        """Process the cars based on the current light state, return the number of vehicles passed this step"""
        passed_this_step = 0
        for direction in range(4):
            is_light_green_for_ns = self.light_state == 0
            is_light_green_for_ew = not is_light_green_for_ns
            is_current_direction_ns = direction in [0, 1]
            is_current_direction_ew = not is_current_direction_ns
            is_light_green_for_this_direction = (
                is_light_green_for_ns and is_current_direction_ns
            ) or (is_light_green_for_ew and is_current_direction_ew)

            if is_light_green_for_this_direction:
                if self.wait_times_in_directions[direction]:
                    self.wait_times_in_directions[direction].popleft()
                    passed_this_step += 1
        self.cumulative_vehicles_passed += passed_this_step
        return passed_this_step

    def _add_new_vehicles(self) -> int:
        """Add new vehicles to the queues, return the number of vehicles arrived this step"""
        arrived_this_step = 0
        for direction in range(4):
            if self.random_generator.random() < self.arrival_prob:
                self.wait_times_in_directions[direction].append(0)
                arrived_this_step += 1
        return arrived_this_step

    def _update_waiting_times(self) -> int:
        """Update the waiting times of vehicles in the queues, return the total waited time this step"""
        waited_this_step = 0
        for direction in range(4):
            for car_i in range(len(self.wait_times_in_directions[direction])):
                self.wait_times_in_directions[direction][car_i] += 1
                waited_this_step += 1
        self.cumulative_time_waited += waited_this_step
        return waited_this_step

    @property
    def wait_times_per_each_car_per_direction(self) -> list[list[int]]:
        return [list(q) for q in self.wait_times_in_directions]

    def reset(self) -> None:
        self.__init__(
            arrival_prob=self.arrival_prob,
            random_seed=self.random_seed,
            logger=self.logger,
        )
