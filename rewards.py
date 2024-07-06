from simulation import StepResult


def reward_based_on_vehicles_in_queue(step_result: StepResult) -> float:
    return -sum(len(wait_times) for wait_times in step_result.wait_times_in_directions)


def reward_based_on_passed_vehicles(step_result: StepResult) -> float:
    return step_result.passed_vehicles


def reward_based_on_max_wait_time(step_result: StepResult) -> float:
    max_per_direction = [
        max(wait_times) if wait_times else 0
        for wait_times in step_result.wait_times_in_directions
    ]
    return -max(max_per_direction)


def reward_based_on_avg_wait_time(step_result: StepResult) -> float:
    total_wait_time = sum(
        sum(wait_times) for wait_times in step_result.wait_times_in_directions
    )
    total_vehicles = sum(
        len(wait_times) for wait_times in step_result.wait_times_in_directions
    )
    return -total_wait_time / total_vehicles if total_vehicles > 0 else 0
