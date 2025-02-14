import numpy as np
import matplotlib.pyplot as plt
from particle_swarm_optimizer import ParticleSwarmOptimizer, PenaltyMethod, LogLevel

# Problem parameters
h = 1  # Height of the starting point
n = 30 # Number of points (including the fixed points)
independent_runs = 10
bounds = (0, 1)  # Bounds for y_i

# Define the objective function for the Brachistochrone problem
def brachistochrone_objective(y: np.ndarray) -> float:
    """
    Compute the total travel time for the Brachistochrone problem.
    Args:
        y: Array of vertical coordinates [y2, y3, ..., y_(n-1)] (n-2 variables).
    Returns:
        Total travel time T.
    """
    y_full = np.concatenate(([1], y, [0]))  # Add fixed points y1 = 1 and yn = 0
    n = len(y_full)
    x = np.linspace(0, 1, n)  # Uniform x-coordinates
    dx = x[1] - x[0]  # Delta x

    T = 0
    for i in range(n - 1):
        dy = y_full[i + 1] - y_full[i]
        ds = np.sqrt(dx**2 + dy**2)  # Segment length
        v1 = np.sqrt(h - y_full[i])  # Speed at point i
        v2 = np.sqrt(h - y_full[i + 1])  # Speed at point i+1
        T += ds / (v1 + v2)  # Add segment travel time

    return T

# Analytical solution
def analytical_solution(n: int) -> tuple:
    """
    Compute the analytical solution for the frictionless Brachistochrone problem.
    Args:
        n: Number of points along the trajectory.
    Returns:
        x_analytical: Array of x-coordinates.
        y_analytical: Array of y-coordinates.
    """
    a = 0.572917  # Constant derived for the analytical solution
    theta = np.linspace(0, 2.412, n)  # Uniformly spaced theta values
    x_analytical = a * (theta - np.sin(theta))
    y_analytical = -a * (1 - np.cos(theta)) + 1
    return x_analytical, y_analytical

# Comparison function
def compare_solutions(n: int, numerical_y: np.ndarray):
    """
    Compare the numerical solution obtained via PSO with the analytical solution.
    Args:
        n: Number of points along the trajectory.
        numerical_y: Array of y-coordinates for the numerical solution.
    """
    # Analytical solution
    x_analytical, y_analytical = analytical_solution(n)

    # Numerical solution
    x_numerical = np.linspace(0, 1, n)
    y_numerical = np.concatenate(([1], numerical_y, [0]))  # Add fixed points y1 = 1 and yn = 0

    # Compute the differences
    y_difference = y_numerical - np.interp(x_numerical, x_analytical, y_analytical)

   # Plot comparison
    plt.figure(figsize=(8, 6))
    plt.plot(x_analytical, y_analytical, label="Analytical Solution", linewidth=2, color='blue')
    plt.plot(x_numerical, y_numerical, 'o-', label="Numerical Solution (PSO)", markersize=6, color='orange')
    plt.xlabel(r"$x$", fontsize=12)
    plt.ylabel(r"$y$", fontsize=12)
    # plt.title(f"Brachistochrone Problem: Comparison for $n = {n}$", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True)

    # Save the figure
    filepath = f"best_plots/P4_comparison_n{n}.png"
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"Comparison plot saved to {filepath}")

    # Print comparison statistics
    max_difference = np.max(np.abs(y_difference))
    print(f"Maximum difference between analytical and numerical solutions for n = {n}: {max_difference:.6f}")

if __name__ == '__main__':
    # Hyperparameters for the optimizer
    n_particles = 59
    w = 0.406
    c1 = 1.527
    c2 = 2.135
    static_penalty = 7535

    print(f"Solving Brachistochrone problem for n = {n}...")
    n_dimensions = n - 2  # Number of design variables

    # Set up the static penalty method
    penalty_method = PenaltyMethod.STATIC
    penalty_method.set_static_penalty(static_penalty)

    # Perform independent runs and store the results
    overall_best_solution = None
    overall_best_value = float('inf')
    histories = []

    for run in range(independent_runs):
        print(f"Run {run + 1}/{independent_runs}")

        pso = ParticleSwarmOptimizer(
            objective_func=brachistochrone_objective,
            inequality_constraints=[],  # No inequality constraints
            equality_constraints=[],  # No equality constraints
            n_dimensions=n_dimensions,
            bounds=bounds,
            n_particles=n_particles,
            w=w,
            c1=c1,
            c2=c2,
            penalty_method=penalty_method,
            log_level=LogLevel.NONE
        )

        # Optimize
        best_solution, best_value, history = pso.optimize(n_iterations=1500, tol=1e-5, patience=50)
        histories.append(history)

        # Update the overall best solution
        if best_value < overall_best_value:
            overall_best_solution = best_solution
            overall_best_value = best_value

    # Print the best solution and value
    print(f"Best solution for n = {n}:", np.concatenate(([1], overall_best_solution, [0])))
    print(f"Best objective value for n = {n}:", overall_best_value)

    # Compare the numerical solution with the analytical solution
    compare_solutions(n, overall_best_solution)

    # Plot the convergence trends
    filepath = pso.plot_convergence(histories, title=f"P4_{n}D", output_dir="best_plots")
    print(f"Convergence plot P4 n={n} saved to {filepath}")


