# traffic-lights
My experiment with creating a DQN model to control a 4-way traffic intersection.
State space:
1) The state of the traffic light (0 for NS, 1 for EW)
2) Is the traffic light transitioning to the state?
3) Queue lengths in direction N
4) Queue lengths in direction S
5) Queue lengths in direction E
6) Queue lengths in direction W
7) Average wait times in direction N
8) Average wait times in direction S
9) Average wait times in direction E
10) Average wait times in direction W

In short, the DQN model shaves off a few seconds for the average waiting time under a timer-based agent.

To view the results, launch main.py, which will first train and evaluate the DQN model, and then do the same for a timer-based model, while plotting the results and printing the average waiting times for the intersections under their control

The code has some potential to improve upon it further. For example, by accounting for the fact that when the green light comes on - the car's won't immediately start moving at a constant rate. More models and other agents could be added as well. MLFlow could be used to track experiments
