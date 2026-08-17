import os

import optuna
import plotly.io as pio

from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_slice
from optuna.visualization import plot_contour
from optuna.visualization import plot_parallel_coordinate


def analyze_hpo_results(storage_url, study_name="scratchnet", save_dir="reports"):
    """
    Loads a study from the DB and generates Plotly visualizations.
    Saves interactive HTML files to ensure results are viewable even if notebook rendering fails.
    """
    from IPython.display import display, IFrame

    # 1. Setup Environment
    os.makedirs(save_dir, exist_ok=True)

    # Try to set a robust renderer for notebooks
    try:
        # 'iframe' is usually safest for Kaggle/Colab
        pio.renderers.default = "iframe"
    except:
        print(f"Failed to change Plotly default renderer")

    print(f"Loading study '{study_name}' from {storage_url}...")
    try:
        study = optuna.load_study(study_name=study_name, storage=storage_url)
    except Exception as e:
        print(f"Failed to load study: {e}")
        return

    if len(study.trials) == 0:
        print("Study has no trials.")
        return

    print(f"Total Trials: {len(study.trials)}")
    print(f"Best Value (AR-MAE): {study.best_value:.6f}")
    print("Best Params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # Helper to save and show
    def process_plot(fig, filename, title):
        fig.update_layout(title=title)

        # Save HTML (Always works, can be downloaded and opened in browser)
        save_path = os.path.join(save_dir, filename)
        fig.write_html(save_path)
        print(f"Saved: {save_path}")

        # Try interactive show (might fail in some envs)
        # try:
        #     fig.show()
        # except Exception:
        #     print("Interactive plot failed to render inline (check saved HTML).")
        # Explicitly display the saved file in an IFrame
        # This bypasses Plotly's internal temp file caching
        try:
            display(IFrame(src=save_path, width="100%", height="500px"))
        except Exception as e:
            print(f"Could not display IFrame: {e}")

    # 1. Optimization History
    # Goal: See if the HPO actually found better models over time or if it flatlined.
    # If the curve goes down, HPO worked
    try:
        fig_hist = plot_optimization_history(study)
        process_plot(fig_hist, "history.html", "1. Optimization History (Convergence)")
    except Exception as e: print(f"Plot failed: {e}")

    # 2. Hyperparameter Importance
    # Goal: Identify which params drive performance.
    # For the paper: "We found that L1 weight and Learning Rate were the most critical..."
    try:
        if len(study.trials) > 1:
            fig_imp = plot_param_importances(study)
            # process_plot(fig, "importance.html", "Hyperparameter Importance")
            process_plot(fig_imp, "importance.html", "2. Hyperparameter Importance")
    except Exception as e:
        print(f"Importance plot failed: {e}")

    # 3. Slice Plot
    # Goal: See the individual relationship of each param to the error.
    # Look for "U-shapes" (optimum in middle) or "Linear slopes" (optimum at edge)
    try:
        fig_slice = plot_slice(study)
        # process_plot(fig, "slices.html", "Parameter Slices")
        process_plot(fig_slice, "slices.html", "3. Individual Parameter Slices")
    except Exception as e: print(f"Slice plot failed: {e}")

    # 4. Contour Plots (Pairwise Interactions)
    # Goal: See how two heavy-hitters interact.
    # Specifically useful for: L1 vs Spectral, or LR vs Batch Size

    # Let's pick the top 3 most important params to plot contours for
    try:
        # Get importance to find top params
        try:
            importance = optuna.importance.get_param_importances(study)
            top_params = list(importance.keys())[:3]
        except:
            # Fallback if importance calc fails (too few trials)
            top_params = list(study.best_params.keys())[:2]

        if len(top_params) >= 2:
            fig_land = plot_contour(study, params=top_params)
            # process_plot(fig, "contour.html", f"Contour Landscape ({top_params})")
            process_plot(fig_land, "contour.html", f"4. Contour Landscape ({top_params})")
    except Exception as e: print(f"Contour plot failed: {e}")

    # 5. Parallel Coordinates
    # Goal: Visualizing the "flow" of high-dimensional configs.
    # Useful to spot clusters of good runs (e.g., "All good runs have High L1 and Low LR")
    try:
        fig_para = plot_parallel_coordinate(study)
        # process_plot(fig, "parallel.html", "High-Dimensional Overview")
        process_plot(fig_para, "parallel.html", "5. Parallel Coordinates (High-Dim View)")
    except Exception as e: print(f"Parallel plot failed: {e}")

    print(f"\n✅ Analysis complete. All interactive plots saved to '{save_dir}/'")
