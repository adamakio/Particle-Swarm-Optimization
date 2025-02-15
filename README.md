# Assignment 1 - AER1415

## Description
This repository contains the code for Assignment 1 of the AER1415 course. This code implements the Particle Swarm Optimization algorithm and is tested on multiple case studies. The hyperparameters are optimized on the 2D bump test function and applied to the remaining problems.

## Prerequisites
Before running the code, ensure you have the following software installed:
- Python 3.9.18
- Required Python packages (listed in `requirements.txt`)

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/adamakio/Particle-Swarm-Optimization.git
    ```
2. Navigate to the project directory:
    ```bash
    cd Particle-Swarm-Optimization
    ```
3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Running Specific Scripts
- To obtain the optimal hyperparameters, run:
    ```bash
    python bump_test_function_optimize.py
    ```
- To run P1, set `n_dimensions` in the script `rosenbrock_test_function.py` to 2 or 5 and execute:
    ```bash
    python rosenbrock_test_function.py
    ```
- To run P2, execute:
    ```bash
    python p2_case_study.py
    ```
- To run P3 with different dimensions, change `n_dimensions` in the script `bump_test_function.py` to 2, 10, or 50 and execute:
    ```bash
    python bump_test_function.py
    ```
- To run P4, set `n` in the script `brachistochrone_problem.py` to `n=15` or `n=30` and execute:
    ```
    ```bash
    python brachistochrone_problem.py
    ```
- To run P5, execute:
    ```bash
    python inverse_problem.py
    ```

## Bonus Question
The answer to the bonus question is found in `best_solution_P3_n50.txt` and can be verified by executing:
    ```bash
    python verify_P3_n50.py
    ```

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Contact
For any questions or issues, please contact adamhamaimou@gmail.com.
