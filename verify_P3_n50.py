import numpy as np
from bump_test_function import bump_test_function, inequality_constraint1, inequality_constraint2

if __name__ == '__main__':
    # Load the solution from the ASCII file
    input_filename = "best_solution_P3_n50.txt"

    try:
        with open(input_filename, "r") as file:
            lines = file.readlines()
        
        # Extract the best solution (x*)
        start_index = lines.index("Best solution (x*):\n") + 1
        x_star = np.loadtxt(lines[start_index:start_index + 50])  # Adjust for n=50
        print(f"Best solution (x*): {x_star}")

        # Evaluate the function and constraints
        f_value = bump_test_function(x_star)
        constraint1_value = inequality_constraint1(x_star)
        constraint2_value = inequality_constraint2(x_star)

        # Print verification results
        print("Verification of the best solution:")
        print(f"Objective function value: {f_value}")
        print(f"Constraint 1 (should be < 0): {constraint1_value}")
        print(f"Constraint 2 (should be < 0): {constraint2_value}")

        if constraint1_value < 0 and constraint2_value < 0:
            print("Solution is feasible.")
        else:
            print("Solution is NOT feasible.")

    except FileNotFoundError:
        print(f"Error: The file {input_filename} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")