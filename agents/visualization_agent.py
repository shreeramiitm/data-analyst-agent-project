# agents/visualization_agent.py
# This worker agent is specialized in creating a variety of data visualizations.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

def run(df: pd.DataFrame, params: dict) -> str:
    """
    Entry point for the VisualizationAgent.
    It generates a plot based on the provided DataFrame and parameters.
    Supports scatter, bar, line, and histogram plots.
    Returns a base64 encoded data URI for the generated image.
    """
    print("VisualizationAgent: Running...")

    # Extract parameters from the plan provided by the orchestrator
    plot_type = params.get("plot_type")
    x_col = params.get("x_column")
    y_col = params.get("y_column") # y_col is optional for histograms

    if not plot_type or not x_col:
        raise ValueError("VisualizationAgent requires at least a plot_type and x_column.")

    if x_col not in df.columns:
        raise ValueError(f"X-axis column '{x_col}' not found in the DataFrame. Available columns: {df.columns.tolist()}")
    if y_col and y_col not in df.columns:
         raise ValueError(f"Y-axis column '{y_col}' not found in the DataFrame. Available columns: {df.columns.tolist()}")

    plt.figure(figsize=(12, 8)) # Create a new figure for each plot

    # --- Plotting Logic ---
    if plot_type == "scatter":
        if not y_col: raise ValueError("Scatter plot requires both x_column and y_column.")
        sns.scatterplot(data=df, x=x_col, y=y_col, alpha=0.7)
        if params.get("regression_line"):
            reg_color = params.get("color", "red")
            reg_linestyle = '--' if params.get("linestyle") == "dotted" else '-'
            sns.regplot(data=df, x=x_col, y=y_col, scatter=False, color=reg_color, line_kws={'linestyle': reg_linestyle})

    elif plot_type == "bar":
        if not y_col: raise ValueError("Bar chart requires both x_column and y_column.")
        sns.barplot(data=df, x=x_col, y=y_col)

    elif plot_type == "line":
        if not y_col: raise ValueError("Line chart requires both x_column and y_column.")
        sns.lineplot(data=df, x=x_col, y=y_col)

    elif plot_type == "histogram":
        sns.histplot(data=df, x=x_col, kde=True) # kde adds a density curve
        # Y-column is not used for a standard histogram, the count is automatic
        y_col = "Frequency" # For labeling purposes

    else:
        raise ValueError(f"Plot type '{plot_type}' is not supported. Supported types: scatter, bar, line, histogram.")

    # --- Aesthetics and Formatting ---
    title = f'{plot_type.capitalize()} of {x_col}'
    if y_col and plot_type != 'histogram':
        title += f' vs. {y_col}'

    plt.title(title, fontsize=16)
    plt.xlabel(x_col, fontsize=12)
    plt.ylabel(y_col, fontsize=12)
    plt.xticks(rotation=45) # Rotate x-axis labels to prevent overlap
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    # --- Save plot to in-memory buffer ---
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    print(f"VisualizationAgent: Success. Returning base64 encoded {plot_type} chart.")
    return f"data:image/png;base64,{image_base64}"