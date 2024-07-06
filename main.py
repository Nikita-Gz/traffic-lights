from simulation import TrafficIntersection
from rewards import reward_based_on_avg_wait_time
from evaluation import evaluate_agent
from agents import TimeBasedAgent
from plotting import plot_evaluation_stats


if __name__ == "__main__":
    env = TrafficIntersection()
    agent = TimeBasedAgent()
    result = evaluate_agent(
        env=env,
        agent=agent,
        reward_function=reward_based_on_avg_wait_time,
        num_steps=1000,
    )
    plot_evaluation_stats(result)
