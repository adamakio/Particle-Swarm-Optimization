import numpy as np
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

if __name__ == '__main__':
    # Problem parameters
    n_dimensions = 50 # Change for n=2, 10, 50
    bounds = (0, 10)
    independent_runs = 10

    # Hyperparameters for the optimizer
    n_particles = 59
    w = 0.406
    c1 = 1.527
    c2 = 2.135
    static_penalty = 7535

    # Set up the static penalty method
    penalty_method = PenaltyMethod.STATIC
    penalty_method.set_static_penalty(static_penalty)

    # Perform 10 independent runs and store the results
    overall_best_solution = None
    overall_best_value = float('inf')
    overall_best_constraint1 = None
    overall_best_constraint2 = None
    histories = []

    for run in range(independent_runs):
        print(f"Run {run + 1}/{independent_runs}")

        pso = ParticleSwarmOptimizer(
            objective_func=bump_test_function,
            inequality_constraints=[inequality_constraint1, inequality_constraint2],
            equality_constraints=[],  # No equality constraints in P3
            n_dimensions=n_dimensions,
            bounds=bounds,
            n_particles=n_particles,
            w=w,
            c1=c1,
            c2=c2,
            penalty_method=penalty_method,
            log_level=LogLevel.INFO
        )

        # Optimize
        best_solution, best_value, history = pso.optimize(n_iterations=1000, tol=1e-5, patience=25)
        histories.append(history)

        # Check if the solution is feasible
        inequality_constraint1_value = inequality_constraint1(best_solution)
        inequality_constraint2_value = inequality_constraint2(best_solution)
        is_feasible = inequality_constraint1_value < 0 and inequality_constraint2_value <= 0
        if is_feasible and best_value < overall_best_value:
            overall_best_solution = best_solution
            overall_best_value = best_value
            overall_best_constraint1 = inequality_constraint1_value
            overall_best_constraint2 = inequality_constraint2_value

    # Print the best solution and value
    print("Best solution:", overall_best_solution)
    print("Best objective value:", overall_best_value)
    print("Best constraint 1 value:", overall_best_constraint1)
    print("Best constraint 2 value:", overall_best_constraint2)

    # Plot the convergence trends
    filepath = pso.plot_convergence(histories, title=f"P3_{n_dimensions}D", output_dir="best_plots")
    print(f"Convergence plot P3 n={n_dimensions} saved to {filepath}")

    if n_dimensions == 50:
        # Save the best solution and corresponding x* values to an ASCII file
        output_filename = "best_solution_P3_n50_2.txt"

        with open(output_filename, "w") as file:
            file.write("Best solution found for P3 with n=50\n")
            file.write("====================================\n")
            file.write(f"Best objective value: {overall_best_value}\n")
            file.write(f"Best solution (x*):\n")
            np.savetxt(file, overall_best_solution, fmt="%.6f")  # Save x* values with 6 decimal precision
            file.write(f"\nConstraint 1 value: {overall_best_constraint1}\n")
            file.write(f"Constraint 2 value: {overall_best_constraint2}\n")

        print(f"Best solution saved to {output_filename}")