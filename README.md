# Infinite-Horizon-Stochastic-Optimal-Control

## Project Overview

This project focuses on safe trajectory tracking for a ground differential-drive robot. The robot's objective is to track a desired reference position trajectory while avoiding collisions with obstacles in the environment. The location and orientation of the robot are controlled by linear and angular velocities. The trajectory tracking problem is formulated in the project as a discounted infinite-horizon stochastic optimal control problem. The approaches of receding-horizon certainty equivalent control (CEC) and generalized policy iteration (GPI) are compared. GPI discretizes the state and control spaces and solves the problem directly, whereas CEC approximates the problem as a deterministic optimal control problem. The project report includes a full description of the problem formulation and potential solutions.

## Features
- Trajectory tracking for ground robots.
- Stochastic optimal control formulation.
- Comparison of RHCEC (Receding Horizon Certainty Equivalent Control) and GPI (Global Policy Iteration) approaches.
- Robust collision avoidance.
- Real-time decision-making.
- Flexible parameter configuration.
- Parallelized value iteration for efficiency.
- Practical control strategies considering system noise.

## Project File Structure

### Source Code

#### Necessary Python Libraries

The third party modules used are as listed below. They are included as [`requirements.txt`](Code/requirements.txt).

- numpy
- matplotlib
- casadi
- scipy

Python files

- [utils.py](Code/utils.py) - This file contains code to visualize the desired trajectory, robot's trajectory, and obstacles.
- [main_CEC.py](Code/main_CEC.py) and [main_GPI.py](Code/main_GPI.py) - These files contain the main function that calls the CEC / GPI controller 
- [GPI_class.py](Code/GPI_class.py) - This file contains a class implementing the Global Policy Iteration algorithm
- [RHCEC.py](Code/RHCEC.py) - This file contains the implementation of the Receding Horizon Certainty Equivalent Control (RHCEC) 

## How to run the code

## RHCEC

1. Run the [main_CEC.py](Code/main_CEC.py) Python file.
   - This script executes the Receding Horizon Certainty Equivalent Control (RHCEC) approach for trajectory tracking.
   - Hyperparameters like Q, R, q, and T can be adjusted inside the [RHCEC.py](Code/RHCEC.py) Python file to fine-tune the control behavior.

   - In the terminal, use the following command:
     ```bash
     python main_CEC.py
     ```

## GPI

1. Run the [main_GPI.py](Code/main_GPI.py) Python file.
   - This script runs the Generalized Policy Iteration (GPI) approach for trajectory tracking.
   - You can configure Hyperparameters Q, R, and q inside the `generate_stage_cost` function located in the [GPI_class.py](Code/GPI_class.py) Python file.

   - In the terminal, use the following command:
    ```bash
    python main_GPI.py
    ```

## Results

The performance analysis of the Receding Horizon Certainty Equivalent Control (CEC) algorithm revealed that it can effectively track reference trajectories in a noise-free environment but faces challenges in the presence of motion model noise. Hyperparameter tuning, especially the planning horizon (T), significantly impacts computation time, while adjustments to the Q and R matrices can improve trajectory tracking.

On the other hand, the Generalized Policy Iteration (GPI) algorithm demonstrated its effectiveness in tracking trajectories, but its performance relies heavily on the resolution of the state space and careful selection of hyperparameters. Higher resolutions provide better tracking but increase computational complexity. Striking the right balance between these factors is crucial for achieving optimal tracking results.

These results highlight the strengths and considerations of both algorithms in trajectory tracking tasks.


|RHCEC|GPI|
|---|---|
|<img src="./Results/C10.png" height="200">|<img src="./Results/C1_GPI.png" height="200">|


## Conclusion

In conclusion, this project investigates optimal control algorithms, CEC and GPI, for safe trajectory tracking in a two-dimensional (two-dimensional) environment with obstacles. The results show that the CEC algorithm operates well when there is no system noise but suffers from accurate tracking when there is noise. It is fast and computationally efficient, although it is sensitive to the time horizon (T) chosen. Choosing an extremely low or extremely high T causes tracking issues or algorithm crashes, accordingly. The GPI algorithm, on the other hand, is offline and computationally slower, with its performance influenced by state discretization. More states boost performance while increasing calculation time and memory consumption. GPI is less susceptible to noise, but it is still influenced by the approximation of real values to discretized ones.

**Contributing**

Contributions to this project are welcome! If you have any suggestions, find issues, or want to contribute code, please feel free to open an issue or submit a pull request.

s

