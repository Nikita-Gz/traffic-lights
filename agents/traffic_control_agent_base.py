from typing import Literal

from simulation import TrafficIntersection, SimulationStatesEnum


class TrafficControlAgent:
    def get_best_action(self, env: TrafficIntersection) -> Literal[0, 1]:
        """Returns the best action to take based on the current state of the environment"""
        raise NotImplementedError

    # Idk what else to add for now
