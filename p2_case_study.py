import numpy as np
from particle_swarm_optimizer import ParticleSwarmOptimizer, PenaltyMethod, LogLevel

# Define the objective function for P2
def p2_objective_function(x: np.ndarray) -> float:
    return x[0]**2 + 0.5 * x[0] + 3 * x[0] * x[1] + 5 * x[1]**2

# Define the inequality constraints for P2
def inequality_constraint1(x: np.ndarray) -> float:
    return 3 * x[0] + 2 * x[1] + 2

def inequality_constraint2(x: np.ndarray) -> float:
    return 15 * x[0] - 3 * x[1] - 1

# Problem parameters
n_dimensions = 2
bounds = (-1, 1)
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
        objective_func=p2_objective_function,
        inequality_constraints=[inequality_constraint1, inequality_constraint2],
        equality_constraints=[],  # No equality constraints in P2
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
    best_solution, best_value, history = pso.optimize(n_iterations=200, tol=1e-5, patience=25)
    histories.append(history)
    
    # Check if the solution is feasible
    is_feasible = all(g(best_solution) <= 0 for g in [inequality_constraint1, inequality_constraint2])
    if is_feasible and best_value < overall_best_value:
        overall_best_solution = best_solution
        overall_best_value = best_value

# Print the best solution and value
print("Best solution:", overall_best_solution)
print("Best objective value:", overall_best_value)

# Plot the convergence trends
filepath = pso.plot_convergence(histories, title=f"P2_{n_dimensions}D", output_dir="best_plots")
print(f"Convergence plot for P2 saved to {filepath}")