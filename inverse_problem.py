import numpy as np
import matplotlib.pyplot as plt
from particle_swarm_optimizer import ParticleSwarmOptimizer, PenaltyMethod, LogLevel

# Load MeasuredResponse.dat
def load_measured_response(filepath):
    """
    Load the measured response from a file.
    Args:
        filepath: Path to the data file.
    Returns:
        time: Array of time instances.
        u_measured: Array of measured displacements.
    """
    data = np.loadtxt(filepath)
    time = data[:, 0]
    u_measured = data[:, 1]
    return time, u_measured

# Define the analytical solution for the damped linear oscillator
def analytical_solution(params, time):
    """
    Calculate the analytical displacement of the system.
    Args:
        params: Array containing [m, k, c, omega, F0].
        time: Array of time instances.
    Returns:
        u: Analytical displacement at each time instance.
    """
    m, k, c, omega, F0 = params

    if k / m - (c / (2 * m))**2 < 0:
        # Return None if the system is not stable
        return None
    
    C = np.sqrt((k - m * omega**2)**2 + (c * omega)**2)
    alpha = np.arctan(c * omega / (k - m * omega**2))
    omegad = np.sqrt(k / m - (c / (2 * m))**2)
    A = -(F0 / C) * np.cos(alpha)
    B = -(F0 / (C * omegad)) * (omega * np.sin(alpha) + c / (2 * m) * np.cos(alpha))
    
    u = np.array([
        (A * np.cos(omegad * t) + B * np.sin(omegad * t)) * np.exp(-c * t / (2 * m))
        + (F0 / C) * np.cos(omega * t - alpha)
        for t in time
    ])
    return u

# Define the objective function for optimization
def objective_function(x):
    """
    Compute the error between measured and analytical responses.
    Args:
        x: Array containing [k, c].
    Returns:
        Error as the sum of squared differences.
    """
    m = 1.0  # Mass (known)
    omega = 0.1  # Forcing frequency (known)
    F0 = 1.0  # Amplitude of the forcing (known)
    
    # Full parameter vector
    params = [m, x[0], x[1], omega, F0]
    
    # Analytical response
    u_analytical = analytical_solution(params, time)
    if u_analytical is None:
        # Return infinity if the analytical solution is invalid
        return np.inf
    
    # Compute the error
    error = np.sum((u_measured - u_analytical)**2)
    return error

if __name__ == '__main__':
    # Load measured response
    time, u_measured = load_measured_response("MeasuredResponse.dat")

    # PSO Hyperparameters
    n_particles = 59
    w = 0.406
    c1 = 1.527
    c2 = 2.135
    static_penalty = 7535

    # Set up the PSO optimizer
    penalty_method = PenaltyMethod.STATIC
    penalty_method.set_static_penalty(static_penalty)

    # Define bounds for [k, c]
    bounds = [(0.1, 100), (0.01, 10)]  # k in [0.1, 100], c in [0.01, 10]
    n_dimensions = len(bounds)

    # Perform the optimization
    overall_best_solution = None
    overall_best_value = float('inf')
    independent_runs = 10
    histories = []

    for run in range(independent_runs):
        print(f"Run {run + 1}/{independent_runs}")
        
        pso = ParticleSwarmOptimizer(
            objective_func=objective_function,
            inequality_constraints=[],  # No inequality constraints (they are handled by returning np.inf)
            equality_constraints=[],  # No equality constraints
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
        best_solution, best_value, history = pso.optimize(n_iterations=500, tol=1e-5, patience=50)
        histories.append(history)
        
        if best_value < overall_best_value:
            overall_best_solution = best_solution
            overall_best_value = best_value

    # Output the results
    print("Best solution (k, c):", overall_best_solution)
    print("Best objective value:", overall_best_value)

    # Plot the measured vs analytical response for the best solution
    best_params = [1.0, overall_best_solution[0], overall_best_solution[1], 0.1, 1.0]
    u_best = analytical_solution(best_params, time)

    plt.figure(figsize=(10, 6))
    plt.plot(time, u_measured, label="Measured Response", color="blue", linewidth=2)
    plt.plot(time, u_best, label="Best Analytical Response", color="red", linestyle="--")
    plt.xlabel("Time (s)")
    plt.ylabel("Displacement (u)")
    plt.title("Comparison of Measured and Analytical Responses")
    plt.legend()
    plt.grid(True)
    plt.savefig("best_plots/P5_comparison.png", dpi=300)
    plt.show()

    # Plot convergence trends
    filepath = pso.plot_convergence(histories, title="P5_Optimization", output_dir="best_plots")
    print(f"Convergence plot saved to {filepath}")
