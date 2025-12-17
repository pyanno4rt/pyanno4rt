"""Metrics graph."""

# Author: Tim Ortkamp

# %% External package import

from IPython import get_ipython
from matplotlib.pyplot import get_current_fig_manager, GridSpec, subplots
from numpy import linspace
from seaborn import lineplot, scatterplot

# %% Internal package import

from pyanno4rt.tools import (
    get_machine_learning_constraints, get_machine_learning_objectives)

# %% Set options

try:
    get_ipython().run_line_magic('matplotlib', 'qt5')
except AttributeError:
    pass

# %% Class definition


class MetricsGraph():
    """
    Metrics graph class.

    This class provides a metrics graph for the data-driven models.
    """

    def view(
            self,
            treatment_plan,
            model_name):
        """Open the metrics graph."""

        def create_subtitle(figure, grid, title):
            """Create a row subtitle."""

            # Add the subplot grid
            row = figure.add_subplot(grid)

            # Set the row title
            row.set_title(f"{title}\n", fontweight='semibold', pad=20)

            # Turn the row frame off
            row.set_frame_on(False)

            # Hide the axis
            row.axis('off')

        # Get the model
        model = next((
            model for model in treatment_plan.data_model_handler.models
            if model.label == model_name), None)

        # Get the evaluation metrics data
        data = model.evaluator.results

        # Specify the evaluation modes
        modes = ('Full', 'Cross-validated')

        # Create a figure and subplots
        figure, axis = subplots(
            nrows=2, ncols=3, figsize=(8, 11), squeeze=False)

        # Initialize the row number and the grid layout object
        row_number = 0
        grid = GridSpec(2, 3)

        # Loop over the number of modes
        for j, _ in enumerate(modes):

            # Plot the AUC-ROC points
            scatterplot(
                x="False Positive Rate", y="True Positive Rate",
                data=data['auc_roc'][modes[j]]['curve'], s=50,
                legend=False, ax=axis[row_number, 0])

            # Add the interpolation line to the AUC-ROC plot
            axis[row_number, 0].plot(
                "False Positive Rate", "True Positive Rate",
                data=data['auc_roc'][modes[j]]['curve'], lw=1, color='k')

            # Add the diagonal line to the AUC-ROC plot
            axis[row_number, 0].plot(
                linspace(0, 1, 100), linspace(0, 1, 100), color='k',
                ls='--', lw=1)

            # Fill the region under the interpolation line
            axis[row_number, 0].fill_between(
                y1=data['auc_roc'][modes[j]]['curve']['True Positive Rate'],
                x=data['auc_roc'][modes[j]]['curve']['False Positive Rate'],
                alpha=.3, color='red')

            # Set the plot title
            axis[row_number, 0].set_title(
                "Receiver Operating Characteristic", fontsize=11)

            # Add the AUC-ROC value to the plot
            axis[row_number, 0].annotate(
                r"AUC$=$"f'{round(data["auc_roc"][modes[j]]["value"], 4)}',
                xy=(0.82, 0.03), fontsize=8)

            # Add the plot grid
            axis[row_number, 0].grid()

            # Plot the AUC-PR line
            lineplot(
                x="Recall", y="Precision", data=data['auc_pr'][modes[j]],
                ax=axis[row_number, 1])

            # Set the limits for the y-axis
            axis[row_number, 1].set_ylim(0, 1)

            # Set the plot title
            axis[row_number, 1].set_title(
                "Precision-Recall Curve", fontsize=11)

            # Add the plot grid
            axis[row_number, 1].grid()

            # Plot the F1 line
            data['f1'][modes[j]]['values'].plot(
                ax=axis[row_number, 2], ylim=(0, 1))

            # Set the labels for x- and y-axis
            axis[row_number, 2].set_xlabel("Threshold")
            axis[row_number, 2].set_ylabel("F1 Score")

            # Set the plot title
            axis[row_number, 2].set_title("F1 Curve", fontsize=11)

            # Add a vertical line to indicate the best F1 position
            axis[row_number, 2].axvline(
                data['f1'][modes[j]]['best'], lw=1, ls='--', color='k')

            # Add the plot grid
            axis[row_number, 2].grid()

            # Create the subtitle for the plot row
            create_subtitle(
                figure, grid[row_number, ::], f'{model.label} ({modes[j]})')

            # Increment the row number
            row_number += 1

            # Apply a tight layout
            figure.tight_layout()

            # Get the figure manager
            figure_manager = get_current_fig_manager()

            # Set the window title
            figure_manager.set_window_title("Metrics Graphs")

            # Show the full-screen plot
            figure_manager.window.showMaximized()
