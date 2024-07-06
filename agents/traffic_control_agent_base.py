from typing import Literal

from simulation import TrafficIntersection


class TrafficControlAgent:
    def get_best_action(self, env: TrafficIntersection) -> Literal[0, 1]:
        raise NotImplementedError

    # Idk what else to add for now
