import pandas as pd

if __name__ == "__main__":
    # Load the CSV file (update the path as needed)
    file_path = "best_solutions/best_solutions_P3_2D.csv"
    df = pd.read_csv(file_path)

    # Convert "violated" column to boolean if needed
    df["violated"] = df["violated"].astype(bool)

    # Filter only feasible solutions (violated == False)
    df_feasible = df[df["violated"] == False]

    # Separate by penalty method
    df_static = df_feasible[df_feasible["penalty_method"] == "STATIC"].nsmallest(3, "objective_value")
    df_adaptive = df_feasible[df_feasible["penalty_method"] == "ADAPTIVE"].nsmallest(3, "objective_value")

    # Function to generate LaTeX table code
    def generate_latex_table(df, title):
        
        penalty_param_header = "$\\rho$" if title == "STATIC" else "$C$ & $\\alpha$"
        latex_code = f"\\begin{'{table}'}[h]\n\\centering\n\\caption{{Top 3 Hyperparameter Configurations for {title} Penalty Method}}\n"
        if title == "STATIC":
            latex_code += "\\begin{tabular}{|c|c|c|c|c|c|c|}\n\\hline\n"
        else:
            latex_code += "\\begin{tabular}{|c|c|c|c|c|c|c|c|}\n\\hline\n"
        latex_code += f"Swarm size & $w$ & $c_1$ & $c_2$ & {penalty_param_header} & Best solution & Function value \\\\\n\\hline\n"

        for _, row in df.iterrows():
            # Format best position to 5 significant figures
            best_position = "[" + ", ".join(f"{round(float(x), 5)}" for x in eval(row["best_position"])) + "]"

            # Format penalty parameter
            penalty_param = f"{int(row['rho'])}" if title == "STATIC" else f"{row['C']:.2f} & {row['alpha']:.2f}"
            
            # Add row to LaTeX table
            latex_code += f"{int(row['n_particles'])} & {row['w']:.3f} & {row['c1']:.3f} & {row['c2']:.3f} & {penalty_param} & {best_position} & {row['objective_value']:.5f} \\\\\n"


        latex_code += "\\hline\n\\end{tabular}\n\\end{table}\n"
        return latex_code

    # Generate LaTeX tables
    latex_static = generate_latex_table(df_static, "STATIC")
    latex_adaptive = generate_latex_table(df_adaptive, "ADAPTIVE")

    # Save to files
    with open("latex_static_table.txt", "w") as f:
        f.write(latex_static)

    with open("latex_adaptive_table.txt", "w") as f:
        f.write(latex_adaptive)
