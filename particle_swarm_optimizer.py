"""
particle_swarm_optimizer.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
This file contains the implementation of the Particle Swarm Optimization (PSO) algorithm with penalty methods.
"""

import csv
import numpy as np
from enum import Enum
from typing import Callable, List, Tuple, Union, Optional
import matplotlib.pyplot as plt

class PenaltyMethod(Enum):
    STATIC = "static"
    ADAPTIVE = "adaptive"

    def __init__(self, *args):
        self.C: Optional[float] = None
        self.alpha: Optional[float] = None
        self.static_penalty: Optional[float] = None

    def set_static_penalty(self, penalty: float):
        if self == PenaltyMethod.STATIC:
            self.static_penalty = penalty

    def set_adaptive_params(self, C: float, alpha: float):
        if self == PenaltyMethod.ADAPTIVE:
            self.C = C
            self.alpha = alpha


class LogLevel(Enum):
    NONE = 0
    INFO = 1
    DEBUG = 2

    def __ge__(self, other):
        if isinstance(other, LogLevel):
            return self.value >= other.value
        return NotImplemented


class ParticleSwarmOptimizer:
    def __init__(
        self,
        objective_func: Callable[[np.ndarray], float],
        inequality_constraints: List[Callable[[np.ndarray], float]],
        equality_constraints: List[Callable[[np.ndarray], float]],
        n_dimensions: int,
        bounds: Union[Tuple[float, float], List[Tuple[float, float]]],
        n_particles: int = 30,
        w: float = 0.7,
        c1: float = 1.5,
        c2: float = 2.0,
        penalty_method: PenaltyMethod = PenaltyMethod.STATIC,
        log_level: LogLevel = LogLevel.NONE,
    ):
        """
        Initialize the Particle Swarm Optimization (PSO) algorithm.

        Args:
            objective_func (Callable[[np.ndarray], float]): Objective function to minimize.
            inequality_constraints (List[Callable[[np.ndarray], float]]): List of inequality constraint functions (g(x) <= 0).
            equality_constraints (List[Callable[[np.ndarray], float]]): List of equality constraint functions (h(x) = 0).
            n_dimensions (int): Dimensionality of the search space.
            bounds (Union[Tuple[float, float], List[Tuple[float, float]]]): Bounds for the search space as a tuple or list of tuples.
            n_particles (int): Number of particles in the swarm.
            w (float): Inertia weight.
            c1 (float): Cognitive coefficient.
            c2 (float): Social coefficient.
            penalty_method (PenaltyMethod): Penalty method ('static' or 'adaptive').
            log_level (LogLevel): Logging level (NONE, INFO, DEBUG).
        """
        # Process bounds
        if isinstance(bounds, tuple):
            bounds = [bounds] * n_dimensions

        # Validate inputs
        assert len(bounds) == n_dimensions, "Length of bounds must match the dimensionality."
        assert all(len(b) == 2 for b in bounds), "Each bound must be a tuple of (lower, upper)."
        assert all(b[0] < b[1] for b in bounds), "Lower bound must be less than upper bound for all dimensions."
        assert n_particles > 0, "Number of particles must be greater than zero."

        self.objective_func = objective_func
        self.inequality_constraints = inequality_constraints
        self.equality_constraints = equality_constraints
        self.n_dimensions = n_dimensions
        self.bounds = np.array(bounds)
        self.n_particles = n_particles
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.penalty_method = penalty_method
        self.log_level = log_level

        self.positions, self.velocities = self.initialize_swarm()
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_values = np.array([self.evaluate(p) for p in self.positions])
        self.global_best_position, self.global_best_value = self.update_global_best()

        if self.log_level >= LogLevel.INFO:
            print("Initialized PSO with", n_particles, "particles in a", n_dimensions, "dimensional space.")

    def initialize_swarm(self) -> Tuple[np.ndarray, np.ndarray]:
        """Initialize particles with random positions and velocities."""
        positions = np.array([
            np.random.uniform(bound[0], bound[1], self.n_particles)
            for bound in self.bounds
        ]).T
        velocities = np.array([
            np.random.uniform(-abs(bound[1] - bound[0]), abs(bound[1] - bound[0]), self.n_particles)
            for bound in self.bounds
        ]).T
        return positions, velocities

    def penalty_function(self, x: np.ndarray, iteration: int) -> float:
        """Calculate penalty for constraint violations."""
        inequality_violation = np.sum([max(0, g(x))**2 for g in self.inequality_constraints])
        equality_violation = np.sum([h(x)**2 for h in self.equality_constraints])
        total_violation = inequality_violation + equality_violation

        if self.penalty_method == PenaltyMethod.STATIC:
            return self.penalty_method.static_penalty * total_violation
        elif self.penalty_method == PenaltyMethod.ADAPTIVE:
            C = self.penalty_method.C
            alpha = self.penalty_method.alpha
            return (C * (iteration + 1))**alpha * total_violation
        else:
            raise ValueError(f"Penalty method {self.penalty_method} is not implemented.")

    def evaluate(self, x: np.ndarray, iteration: int = 0) -> float:
        """Evaluate the objective function with penalty for constraints."""
        return self.objective_func(x) + self.penalty_function(x, iteration)

    def update_global_best(self) -> Tuple[np.ndarray, float]:
        """Update the global best position and value."""
        best_idx = np.argmin(self.personal_best_values)
        return self.personal_best_positions[best_idx], self.personal_best_values[best_idx]

    def update_personal_best(self, particle_idx: int, fitness: float):
        """Update the personal best position and value for a particle."""
        if fitness < self.personal_best_values[particle_idx]:
            self.personal_best_positions[particle_idx] = self.positions[particle_idx]
            self.personal_best_values[particle_idx] = fitness

    def update_positions_and_velocities(self):
        """Update the positions and velocities of particles."""
        r1, r2 = np.random.rand(self.n_particles, self.n_dimensions), np.random.rand(self.n_particles, self.n_dimensions)
        cognitive = self.c1 * r1 * (self.personal_best_positions - self.positions)
        social = self.c2 * r2 * (self.global_best_position - self.positions)
        self.velocities = self.w * self.velocities + cognitive + social
        for i in range(self.n_dimensions):
            self.positions[:, i] = np.clip(self.positions[:, i] + self.velocities[:, i], self.bounds[i][0], self.bounds[i][1])

    def optimize(self, n_iterations: int, tol: float = 1e-5, patience: int = 20) -> Tuple[np.ndarray, float, List[float]]:
        """Run the PSO optimization process."""
        history = []
        for t in range(n_iterations):
            if self.log_level >= LogLevel.DEBUG:
                print(f"Iteration {t + 1}/{n_iterations}")

            for i in range(self.n_particles):
                fitness = self.evaluate(self.positions[i], t)
                self.update_personal_best(i, fitness)
                if fitness < self.global_best_value:
                    self.global_best_position = self.positions[i]
                    self.global_best_value = fitness

            self.update_positions_and_velocities()
            history.append(self.global_best_value)

            if self.log_level >= LogLevel.INFO:
                print(f"Iteration {t + 1}: Best value = {self.global_best_value}")

            # Stopping criteria: tolerance for consecutive iterations
            if t > 0 and abs(history[-1] - history[-2]) < tol:
                consecutive_tolerance_count += 1
                if consecutive_tolerance_count >= patience:
                    if self.log_level >= LogLevel.INFO:
                        print("Convergence reached with tolerance threshold.")
                    break
            else:
                consecutive_tolerance_count = 0

        return self.global_best_position, self.global_best_value, history
    
    def plot_convergence(self, all_histories: List[List[float]], title: str, output_dir: str = ".", show_plot: bool = False):
        """Plot all individual runs, mean convergence trend, and save the figure."""
        max_len = max(len(h) for h in all_histories)
        histories_padded = [h + [h[-1]] * (max_len - len(h)) for h in all_histories]
        histories_array = np.array(histories_padded)

        mean_values = np.mean(histories_array, axis=0)
        std_values = np.std(histories_array, axis=0)

        plt.figure(figsize=(12, 8))

        # Plot all individual runs
        for idx, run in enumerate(histories_array):
            plt.plot(run, alpha=0.5, linestyle='--', label=f'Run {idx + 1}')

        # Plot mean and standard deviation
        plt.plot(mean_values, label="Mean Convergence Trend", linewidth=2, color='black')
        plt.fill_between(
            range(max_len),
            mean_values - 3*std_values,
            mean_values + 3*std_values,
            alpha=0.2,
            label="±3 Std. Dev.",
            color='gray',
        )

        plt.xlabel("Iteration")
        plt.ylabel("Objective Value")
        if self.penalty_method == PenaltyMethod.STATIC:
            penalty_params = {"rho": self.penalty_method.static_penalty}
        elif self.penalty_method == PenaltyMethod.ADAPTIVE:
            penalty_params = {"C": self.penalty_method.C, "alpha": self.penalty_method.alpha}
        penalty_params_str = ", ".join(f"{k}={v}" for k, v in penalty_params.items())
        penalty_params_filename = "_".join(f"{k}{v}" for k, v in penalty_params.items())
        full_title = (
            f"Particle Swarm Optimization Convergence: {title} "
            f"n_particles={self.n_particles}, w={self.w}, c1={self.c1}, c2={self.c2}, penalty_method={self.penalty_method.name}, {penalty_params_str}"
        )
        plt.title(full_title)
        plt.legend(loc='best', fontsize=9)
        plt.grid(True)
        plt.savefig(f"{output_dir}/{title}_convergence_n{self.n_particles}_w{self.w}_c1{self.c1}_c2{self.c2}_{self.penalty_method.name}_{penalty_params_filename}.png")
        if show_plot:
            plt.show()

    
    def save_best_solutions(self, best_solutions: List[Tuple[np.ndarray, float]], filename: str = "best_solutions.csv", output_dir: str = ".") -> Optional[dict]:
        """Save the best solutions to a CSV file, appending new runs, and return the best feasible solution."""
        header = [
            "n_particles", "w", "c1", "c2", "penalty_method", "rho", "C", "alpha", "best_position", "objective_value"
        ]
        header += [f"inequality_{i}" for i in range(len(self.inequality_constraints))]
        header += [f"equality_{i}" for i in range(len(self.equality_constraints))]
        header += ["violated"]

        rows = []

        for position, _ in best_solutions:
            constraints_values = [g(position) for g in self.inequality_constraints] + [h(position) for h in self.equality_constraints]
            violated = (
                any(g > 0 for g in constraints_values[:len(self.inequality_constraints)]) 
                or any(abs(h) > 1e-5 for h in constraints_values[len(self.inequality_constraints):])
            )
            penalty_params = [
                self.penalty_method.static_penalty if self.penalty_method == PenaltyMethod.STATIC else "NA",
                self.penalty_method.C if self.penalty_method == PenaltyMethod.ADAPTIVE else "NA", 
                self.penalty_method.alpha if self.penalty_method == PenaltyMethod.ADAPTIVE else "NA"
            ]
            row = [
                self.n_particles, self.w, self.c1, self.c2, self.penalty_method.name
            ]
            row += penalty_params
            row += [
                position.tolist(), self.objective_func(position)
            ]
            row += constraints_values
            row.append(violated)
            rows.append(row)

        with open(f"{output_dir}/{filename}", mode="a", newline="") as file:
            writer = csv.writer(file)
            file.seek(0, 2)
            if file.tell() == 0:  # If file is empty, write the header
                writer.writerow(header)
            writer.writerows(rows)

        # Find and return the best feasible solution
        best_feasible_solution = min(
            (row for row in rows if not row[-1]),
            key=lambda r: r[7],
            default=None
        )

        if best_feasible_solution:
            with open(f"{output_dir}/{filename}", mode="a", newline="") as file:
                writer = csv.writer(file)
                file.seek(0, 2)
                if file.tell() == 0:  # If file is empty, write the header
                    writer.writerow(header)
                writer.writerow(best_feasible_solution)
            return {k: v for k, v in zip(header, best_feasible_solution)}