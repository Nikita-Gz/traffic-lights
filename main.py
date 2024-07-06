from simulation import TrafficIntersection
from rewards import reward_based_on_passed_vehicles
from evaluation import evaluate_agent
from agents import TimeBasedAgent
from plotting import plot_evaluation_stats, plot_car_count_on_directions


if __name__ == "__main__":
    env = TrafficIntersection(
        arrival_prob=0.2,
    )
    agent = TimeBasedAgent(light_duration=60)
    result = evaluate_agent(
        env=env,
        agent=agent,
        reward_function=reward_based_on_passed_vehicles,
        num_steps=1000,
    )
    plot_evaluation_stats(result)
    plot_car_count_on_directions(result)
