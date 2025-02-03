"""
bump_test_function.py
~~~~~~~~~~~~~~~~~~~~~
This file contains the implementation of the bump test function (P3) and its constraints.
The hyperparameters of the Particle Swarm Optimizer are optimized using the 2D bump test function.
"""

import json
import numpy as np
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from particle_swarm_optimizer import ParticleSwarmOptimizer, PenaltyMethod, LogLevel

# Define the objective function and constraints for P3
def bump_test_function(x: np.ndarray) -> float:
    numerator = abs(np.sum(np.cos(x)**4) - 2 * np.prod(np.cos(x)**2))
    denominator = np.sqrt(np.sum([i * (xi**2) for i, xi in enumerate(x, start=1)]))
    return - numerator / denominator

def inequality_constraint1(x: np.ndarray) -> float:
    return 0.75 - np.prod(x)

def inequality_constraint2(x: np.ndarray) -> float:
    return np.sum(x) - (15 * len(x) / 2)

n_dimensions = 2
bounds = (0, 10)
independent_runs = 10

space = [
    Categorical([PenaltyMethod.STATIC, PenaltyMethod.ADAPTIVE], name='penalty_method'),
    Integer(10, 60, name='n_particles'),
    Real(0.4, 1.4, name='w'),
    Real(1.5, 2.0, name='c1'),
    Real(2.0, 2.5, name='c2'),
    Real(1e2, 1e4, name='static_penalty'),
    Real(0.5, 1.0, name='C'),
    Real(1.0, 2.0, name='alpha')
]

@use_named_args(space)
def objective_function(penalty_method, n_particles, w, c1, c2, static_penalty, C, alpha):
    # Set the penalty method parameters
    if penalty_method == PenaltyMethod.STATIC:
        penalty_method.set_static_penalty(static_penalty)
    elif penalty_method == PenaltyMethod.ADAPTIVE:
        penalty_method.set_adaptive_params(C, alpha)

    # Record the overall best value
    overall_best_solution = None
    overall_best_value = None
    histories = []

    # Run the optimizer for multiple independent runs
    for run in range(independent_runs):
        pso = ParticleSwarmOptimizer(
            objective_func=bump_test_function,
            inequality_constraints=[inequality_constraint1, inequality_constraint2],
            equality_constraints=[],
            n_dimensions=n_dimensions,
            bounds=bounds,
            n_particles=n_particles,
            w=w,
            c1=c1,
            c2=c2,
            penalty_method=penalty_method,
            log_level=LogLevel.NONE
        )

        best_solution, best_value, history = pso.optimize(n_iterations=200, tol=1e-5, patience=25)
        histories.append(history)
        if overall_best_solution is None or best_value < overall_best_value:
            overall_best_solution = best_solution
            overall_best_value = best_value

    # Save the best solutions to a CSV file
    best_feasible_solution = pso.save_best_solution(overall_best_solution, filename="best_solutions_P3_2D.csv", output_dir="best_solutions")

    # Plot the histories for all runs
    pso.plot_convergence(histories, title="P3 2D", output_dir="convergence_plots")

    # Return the overall best value
    return best_feasible_solution["objective_value"] if best_feasible_solution is not None else np.inf

def hyperparameter_search():
    res = gp_minimize(objective_function, space, n_calls=200, n_random_starts=20, random_state=42, n_jobs=-1)
    best_hyperparameters = res.x
    print("Best hyperparameters:", best_hyperparameters)

if __name__ == "__main__":
    hyperparameter_search()
