from typing import Literal

from simulation import TrafficIntersection


class TrafficControlAgent:
    def get_best_action(self, env: TrafficIntersection) -> Literal[0, 1]:
        raise NotImplementedError

    # Idk what else to add for now


class TimeBasedAgent(TrafficControlAgent):
    """Switches the traffic light every `light_duration` steps"""

    def __init__(self, light_duration=30):
        self.light_duration = light_duration
        self.steps_since_last_light_change = 0

    def get_best_action(self, env: TrafficIntersection) -> Literal[0, 1]:
        self.steps_since_last_light_change += 1
        if self.steps_since_last_light_change >= self.light_duration:
            self.steps_since_last_light_change = 0
            return 1
        return 0


class DoNothingAgent(TrafficControlAgent):
    """
    Never changes the traffic light
    Drivers on the red light will be very angry
    Drivers on the green light will be very happy
    """

    def get_best_action(self, env: TrafficIntersection) -> Literal[0, 1]:
        return 0
