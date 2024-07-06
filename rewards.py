from simulation import StepResult


def reward_based_on_vehicles_in_queue(step_result: StepResult) -> float:
    return -sum(len(q) for q in step_result.wait_times_in_directions)


def reward_based_on_max_wait_time(step_result: StepResult) -> float:
    max_per_direction = [
        max(q) if q else 0 for q in step_result.wait_times_in_directions
    ]
    return -max(max_per_direction)


def reward_based_on_avg_wait_time(step_result: StepResult) -> float:
    total_wait_time = sum(sum(q) for q in step_result.wait_times_in_directions)
    total_vehicles = sum(len(q) for q in step_result.wait_times_in_directions)
    return -total_wait_time / total_vehicles if total_vehicles > 0 else 0
