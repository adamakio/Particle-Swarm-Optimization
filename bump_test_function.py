import json
import numpy as np
from itertools import product
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

def hyperparameter_search():
    n_dimensions = 2
    bounds = (0, 10)
    independent_runs = 10
    penalty_methods = [PenaltyMethod.STATIC, PenaltyMethod.ADAPTIVE]
    n_particles_values = [10, 35, 60]
    w_values = [0.4, 0.9, 1.4]
    c1_values = [1.5, 1.75, 2.0]
    c2_values = [2.0, 2.25, 2.5]
    static_penalty_values = [1e2, 1e3, 1e4]
    adaptive_params = [(0.5, 1.0), (0.5, 1.5), (0.5, 2.0)]

    best_overall_solution = None

    for penalty_method in penalty_methods:
        for n_particles, w, c1, c2 in product(n_particles_values, w_values, c1_values, c2_values):
            if penalty_method == PenaltyMethod.STATIC:
                for static_penalty in static_penalty_values:
                    penalty_method.set_static_penalty(static_penalty)

                    histories = []
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
                        best_position, best_value, history = pso.optimize(n_iterations=100, tol=1e-6, patience=20)
                        histories.append(history)
                        best_solutions.append((best_position, best_value))

                    # Plot the convergence history
                    pso.plot_convergence(histories, title=f"P3_{n_dimensions}D", output_dir="convergence_plots")

                    # Save best solutions to a file and retrieve the best feasible solution
                    best_solution = pso.save_best_solutions(best_solutions, filename=f"best_solutions_P3_{n_dimensions}D.csv", output_dir="best_solutions")

                    if best_solution:
                        print("Best feasible solution:")
                        print(json.dumps(best_solution, indent=4))
                        if not best_overall_solution or best_solution["objective_value"] < best_overall_solution["objective_value"]:
                            best_overall_solution = best_solution

            elif penalty_method == PenaltyMethod.ADAPTIVE:
                for C, alpha in adaptive_params:
                    penalty_method.set_adaptive_params(C, alpha)                   
                    
                    histories = []
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
                        best_position, best_value, history = pso.optimize(n_iterations=100, tol=1e-6, patience=20)
                        histories.append(history)
                        best_solutions.append((best_position, best_value))

                    # Plot the convergence history
                    pso.plot_convergence(histories, title=f"P3_{n_dimensions}D", output_dir="convergence_plots")

                    # Save best solutions to a file and retrieve the best feasible solution
                    best_solution = pso.save_best_solutions(best_solutions, filename=f"best_solutions_P3_{n_dimensions}D.csv", output_dir="best_solutions")

                    if best_solution:
                        print("Best feasible solution:")
                        print(json.dumps(best_solution, indent=4))
                        if not best_overall_solution or best_solution["objective_value"] < best_overall_solution["objective_value"]:
                            best_overall_solution = best_solution
    
    print("Best hyperparameters:")
    print(best_overall_solution)

if __name__ == "__main__":
    hyperparameter_search()
