import numpy as np
import os
from particle_swarm_optimizer import ParticleSwarmOptimizer, PenaltyMethod, LogLevel

# Define the objective function and constraints
def bump_test_function(x: np.ndarray) -> float:
    numerator = abs(np.sum(np.cos(x)**4) - 2 * np.prod(np.cos(x)**2))
    denominator = np.sqrt(np.sum([i * (xi**2) for i, xi in enumerate(x, start=1)]))
    return - numerator / denominator

def inequality_constraint1(x: np.ndarray) -> float:
    return 0.75 - np.prod(x)

def inequality_constraint2(x: np.ndarray) -> float:
    return np.sum(x) - (15 * len(x) / 2)

# Set hyperparameter values for STATIC method
static_params = {
    "penalty_method": PenaltyMethod.STATIC,
    "n_particles": 59,
    "w": 0.4063759899463055,
    "c1": 1.527326152164539,
    "c2": 2.1348929944802615,
    "static_penalty": 7535.190137383015,
    "C": None,
    "alpha": None
}

# Set hyperparameter values for ADAPTIVE method
adaptive_params = {
    "penalty_method": PenaltyMethod.ADAPTIVE,
    "n_particles": 42,
    "w": 0.6248024703859039,
    "c1": 1.9679646120707424,
    "c2": 2.0006087252924303,
    "static_penalty": None,
    "C": 0.5040335872985765,
    "alpha": 1.938183640589161
}

# Directory to save plots
output_dir = "best_plots"
os.makedirs(output_dir, exist_ok=True)

# Function to regenerate the convergence plot
def regenerate_convergence_plot(params, title):
    penalty_method = params["penalty_method"]
    
    # Set penalty method parameters
    if penalty_method == PenaltyMethod.STATIC:
        penalty_method.set_static_penalty(params["static_penalty"])
    elif penalty_method == PenaltyMethod.ADAPTIVE:
        penalty_method.set_adaptive_params(params["C"], params["alpha"])

    histories = []

    # Run optimizer for multiple independent runs
    independent_runs = 10  # Adjust as necessary
    n_dimensions = 2
    bounds = (0, 10)

    for run in range(independent_runs):
        pso = ParticleSwarmOptimizer(
            objective_func=bump_test_function,
            inequality_constraints=[inequality_constraint1, inequality_constraint2],
            equality_constraints=[],
            n_dimensions=n_dimensions,
            bounds=bounds,
            n_particles=params["n_particles"],
            w=params["w"],
            c1=params["c1"],
            c2=params["c2"],
            penalty_method=penalty_method,
            log_level=LogLevel.NONE
        )

        _, _, history = pso.optimize(n_iterations=200, tol=1e-5, patience=25)
        histories.append(history)

    # Plot convergence
    filepath = pso.plot_convergence(histories, title=title, output_dir=output_dir)
    print(f"Regenerated convergence plot saved to {filepath}")

# Regenerate plots
regenerate_convergence_plot(static_params, "P3_2D_STATIC")
regenerate_convergence_plot(adaptive_params, "P3_2D_ADAPTIVE")

print("Convergence plots have been regenerated and saved to", output_dir)
