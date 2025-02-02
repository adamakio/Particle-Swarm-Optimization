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
    best_solutions = []
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

        if penalty_method == PenaltyMethod.STATIC:
            pso.penalty_method.set_static_penalty(static_penalty)
        elif penalty_method == PenaltyMethod.ADAPTIVE:
            pso.penalty_method.set_adaptive_params(C, alpha)

        best_position, best_value, _ = pso.optimize(n_iterations=100, tol=1e-6, patience=20)
        best_solutions.append((best_position, best_value))

    return pso.save_best_solutions(best_solutions, filename="best_solutions_P3_2D.csv", output_dir="best_solutions")["objective_value"]

def hyperparameter_search():
    res = gp_minimize(objective_function, space, n_calls=50, n_random_starts=10, random_state=42)
    best_hyperparameters = res.x
    print("Best hyperparameters:", best_hyperparameters)

if __name__ == "__main__":
    hyperparameter_search()
