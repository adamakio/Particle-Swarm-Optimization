import numpy as np
import matplotlib.pyplot as plt
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

# Solve the P3 problem
if __name__ == "__main__":
    bounds = (0, 10)  # Example for uniform bounds
    penalty_method = PenaltyMethod.ADAPTIVE
    penalty_method.set_adaptive_params(C=0.5, alpha=1.5)

    independent_runs = 10
    histories = []
    best_solutions = []

    for run in range(independent_runs):
        pso = ParticleSwarmOptimizer(
            objective_func=bump_test_function,
            inequality_constraints=[inequality_constraint1, inequality_constraint2],
            equality_constraints=[],
            n_dimensions=2,  # Change to 10 for n=10
            bounds=bounds,
            n_particles=50,
            penalty_method=penalty_method,
            log_level=LogLevel.INFO,
        )
        best_position, best_value, history = pso.optimize(n_iterations=100, tol=1e-6, patience=20)
        histories.append(history)
        best_solutions.append((best_position, best_value))

    # Plot the convergence history
    pso.plot_convergence(histories, title="P3")

    # Save best solutions to a file and retrieve the best feasible solution
    best_solution = pso.save_best_solutions(best_solutions, filename="best_solutions_P3.csv")

    if best_solution:
        print("Best feasible solution:")
        print(best_solution)