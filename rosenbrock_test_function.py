import numpy as np
from particle_swarm_optimizer import ParticleSwarmOptimizer, PenaltyMethod, LogLevel

# Define the Rosenbrock function for P1
def rosenbrock_function(x: np.ndarray) -> float:
    return sum(100 * (x[i + 1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(len(x) - 1))

# Problem parameters
n_dimensions = 5  # You can also set n_dimensions = 2 for the other case
bounds = (-5, 5)
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
histories = []

for run in range(independent_runs):
    print(f"Run {run + 1}/{independent_runs}")
    
    pso = ParticleSwarmOptimizer(
        objective_func=rosenbrock_function,
        inequality_constraints=[],  # No constraints for P1
        equality_constraints=[],    # No constraints for P1
        n_dimensions=n_dimensions,
        bounds=bounds,
        n_particles=n_particles,
        w=w,
        c1=c1,
        c2=c2,
        penalty_method=penalty_method,
        log_level=LogLevel.DEBUG
    )
    
    # Optimize
    best_solution, best_value, history = pso.optimize(n_iterations=1000, tol=1e-5, patience=25)
    histories.append(history)
    
    # Update the overall best solution
    if best_value < overall_best_value:
        overall_best_solution = best_solution
        overall_best_value = best_value

# Print the best solution and value
print("Best solution:", overall_best_solution)
print("Best objective value:", overall_best_value)

# Plot the convergence trends
filepath = pso.plot_convergence(histories, title=f"P1_{n_dimensions}D", output_dir="best_plots")
print(f"Convergence plot for P1 {n_dimensions}D saved to", filepath)