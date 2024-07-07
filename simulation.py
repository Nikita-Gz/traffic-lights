from dataclasses import dataclass, field
import random
import logging
from collections import deque
from typing import Literal
from enum import Enum

import numpy as np


@dataclass
class StepResult:
    """Stores the results of a step in the environment, not cumulative stats"""

    step_number: int
    new_vehicles: int
    passed_vehicles: int
    additional_time_waited: int
    wait_times_in_directions: list[list[int]] = field(default_factory=list)


class SimulationStatesEnum(Enum):
    LIGHT_STATE = "light_state"
    QUEUE_LENGTHS = "queue_lengths"
    AVG_WAIT_TIMES = "avg_wait_times"


class TrafficIntersection:
    action_size = 2  # 0: Do nothing, 1: Switch the traffic light

    def __init__(
        self,
        arrival_prob: float = 0.1,
        random_seed: int | None = None,
        logger: logging.Logger | None = None,
    ):
        self.arrival_prob = arrival_prob
        self.random_seed = random_seed
        self.logger = logger

        # each item in the queue represents a vehicle and the value represents the time it has waited
        self._wait_times_per_direction: list[deque[int]] = [
            deque()
            for _ in range(4)  # N, S, E, W
        ]

        self.light_state = 0  # 0: N-S Green, 1: E-W Green

        self.step_count = 0
        self.cumulative_time_waited = 0
        self.cumulative_vehicles_passed = 0

        self.random_generator = random.Random(self.random_seed)

    @property
    def wait_times_per_direction(self) -> list[list[int]]:
        """Returns the wait times of vehicles in each direction, but not as deques"""
        return [list(wait_times) for wait_times in self._wait_times_per_direction]

    def _log(self, level: int, message: str):
        if self.logger is not None:
            self.logger.log(level, message)

    def get_state_representation(
        self, states_to_include: list[SimulationStatesEnum]
    ) -> np.ndarray:
        """Return the state representation based on the states to include"""
        state = []
        for state_to_include in states_to_include:
            match state_to_include:
                case SimulationStatesEnum.LIGHT_STATE:
                    state.append(self.light_state)
                case SimulationStatesEnum.QUEUE_LENGTHS:
                    state.extend(
                        [len(queue) for queue in self._wait_times_per_direction]
                    )
                case SimulationStatesEnum.AVG_WAIT_TIMES:
                    state.extend(
                        [
                            sum(queue) / len(queue) if queue else 0
                            for queue in self._wait_times_per_direction
                        ]
                    )
                case _:
                    raise ValueError(f"Unknown state: {state_to_include}")
        return np.array(state)

    def get_state_representation_length(
        self, states_to_include: list[SimulationStatesEnum]
    ) -> int:
        """Return the length of the state representation based on the states to include"""
        return len(self.get_state_representation(states_to_include))

    def step(self, action: Literal[0, 1]) -> StepResult:
        """Take a step in the environment, update the state and return the reward"""

        self._log(logging.DEBUG, f"Step {self.step_count}, action: {action}")

        # Update the state
        self._process_traffic_light(action)
        new_vehicles_this_step = self._add_new_vehicles()
        passed_this_step = self._process_cars_passing()
        waited_this_step = self._update_waiting_times()
        step_result = StepResult(
            step_number=self.step_count,
            new_vehicles=new_vehicles_this_step,
            passed_vehicles=passed_this_step,
            additional_time_waited=waited_this_step,
            wait_times_in_directions=self.wait_times_per_direction,
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
        for direction in range(4):  # N, S, E, W
            is_light_green_for_ns = self.light_state == 0
            is_light_green_for_ew = not is_light_green_for_ns
            is_current_direction_ns = direction in [0, 1]
            is_current_direction_ew = not is_current_direction_ns
            is_light_green_for_this_direction = (
                is_light_green_for_ns and is_current_direction_ns
            ) or (is_light_green_for_ew and is_current_direction_ew)

            if is_light_green_for_this_direction:
                if self._wait_times_per_direction[direction]:
                    self._wait_times_per_direction[direction].popleft()
                    passed_this_step += 1
        self.cumulative_vehicles_passed += passed_this_step
        return passed_this_step

    def _add_new_vehicles(self) -> int:
        """Add new vehicles to the queues, return the number of vehicles arrived this step"""
        arrived_this_step = 0
        for direction in range(4):  # N, S, E, W
            if self.random_generator.random() < self.arrival_prob:
                self._wait_times_per_direction[direction].append(0)
                arrived_this_step += 1
        return arrived_this_step

    def _update_waiting_times(self) -> int:
        """Update the waiting times of vehicles in the queues, return the total waited time this step"""
        waited_this_step = 0
        for direction in range(4):  # N, S, E, W
            for car_i in range(len(self._wait_times_per_direction[direction])):
                self._wait_times_per_direction[direction][car_i] += 1
                waited_this_step += 1
        self.cumulative_time_waited += waited_this_step
        return waited_this_step

    def reset(
        self, new_arrival_prob: float | None = None, new_random_seed: int | None = None
    ):
        self.__init__(
            arrival_prob=self.arrival_prob
            if new_arrival_prob is None
            else new_arrival_prob,
            random_seed=self.random_seed
            if new_random_seed is None
            else new_random_seed,
            logger=self.logger,
        )
