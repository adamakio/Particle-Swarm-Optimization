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

    # Function to return the PNG filename for the top hyperparameter configuration
    def get_best_png_filename(df, title):
        """Returns the PNG filename for the top hyperparameter configuration."""
        if df.empty:
            return None

        # Get the best configuration
        best_row = df.iloc[0]  # Since df is already sorted by objective_value

        # Construct the penalty parameters string
        if best_row["penalty_method"] == "STATIC":
            penalty_params_filename = f"statc{best_row['rho']}"
        else:
            penalty_params_filename = f"C{best_row['C']}_alpha{best_row['alpha']}"

        # Construct the filename
        filename = (
            f"{title}_convergence_n{int(best_row['n_particles'])}_w{best_row['w']}_"
            f"c1{best_row['c1']}_c2{best_row['c2']}_"
            f"{best_row['penalty_method']}_{penalty_params_filename}.png"
        )
        return filename

    # Get filenames for the best hyperparameter configuration of each penalty method
    static_png_filename = get_best_png_filename(df_static, "P3_2D")
    adaptive_png_filename = get_best_png_filename(df_adaptive, "P3_2D")

    # Print the filenames
    print("Best STATIC penalty method PNG filename:", static_png_filename)
    print("Best ADAPTIVE penalty method PNG filename:", adaptive_png_filename)