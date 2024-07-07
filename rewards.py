from simulation import StepResult


def reward_based_on_vehicles_in_queue(step_result: StepResult) -> float:
    return -sum(len(wait_times) for wait_times in step_result.wait_times_in_directions)


def reward_based_on_vehicles_in_red_queue_only(step_result: StepResult) -> float:
    """
    Unlike reward_based_on_vehicles_in_queue, this function will ignore the cars
    that are on the line with the green light, even if the light is transitioning.
    This is done so that the model has an incentive to begin transitioning the light.
    As a note, directions are in the order of [N, S, E, W]
    """
    total_reward = 0
    is_ns_green = step_result.light_state == 0
    for direction_i, wait_times in enumerate(step_result.wait_times_in_directions):
        if (is_ns_green and direction_i in [0, 1]) or (
            not is_ns_green and direction_i in [2, 3]
        ):
            continue
        total_reward -= len(wait_times)
    return total_reward


def reward_based_on_wait_time_in_red_queue_only(step_result: StepResult) -> float:
    """
    Similar to reward_based_on_vehicles_in_red_queue_only, this function will ignore the cars
    that are on the line with the green light, even if the light is transitioning.
    """
    total_reward = 0
    is_ns_green = step_result.light_state == 0
    for direction_i, wait_times in enumerate(step_result.wait_times_in_directions):
        if (is_ns_green and direction_i in [0, 1]) or (
            not is_ns_green and direction_i in [2, 3]
        ):
            continue
        total_reward -= sum(wait_times)
    return total_reward


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
