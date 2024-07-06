import matplotlib.pyplot as plt

from evaluation import EvaluationStats


def plot_evaluation_stats(stats: EvaluationStats):
    """Makes the following plots:
    - Rewards per step
    - Light state per step
    - Number of vehicles waiting per step
    - Average wait time per step
    - Maximum wait time per step
    """
    fig, axs = plt.subplots(3, 2, figsize=(10, 20))  # Adjust the subplot grid to 3x2

    # I hate this repetition, but it's the easiest way to set the titles and labels
    # Maybe turn it into a loop if there is time

    axs[0, 0].plot(stats.rewards)
    axs[0, 0].set_title("Rewards per step")
    axs[0, 0].set_xlabel("Step")
    axs[0, 0].set_ylabel("Reward")

    axs[0, 1].plot(stats.light_states)
    axs[0, 1].set_title("Light state per step")
    axs[0, 1].set_xlabel("Step")
    axs[0, 1].set_ylabel("Light state")

    axs[1, 0].plot(stats.vehicles_waiting)
    axs[1, 0].set_title("Number of vehicles waiting per step")
    axs[1, 0].set_xlabel("Step")
    axs[1, 0].set_ylabel("Number of vehicles")

    axs[1, 1].plot(stats.average_wait_times)
    axs[1, 1].set_title("Average wait time per step")
    axs[1, 1].set_xlabel("Step")
    axs[1, 1].set_ylabel("Average wait time")

    axs[2, 0].plot(stats.max_wait_times)
    axs[2, 0].set_title("Maximum wait time per step")
    axs[2, 0].set_xlabel("Step")
    axs[2, 0].set_ylabel("Max wait time")

    # Hide the unused subplot (bottom right)
    axs[2, 1].axis("off")

    plt.tight_layout()
    plt.show()
