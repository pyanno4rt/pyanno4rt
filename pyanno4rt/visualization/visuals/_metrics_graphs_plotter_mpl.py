"""Data models metrics plot (matplotlib)."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from IPython import get_ipython
from matplotlib.pyplot import get_current_fig_manager, GridSpec, subplots
from numpy import linspace
from seaborn import lineplot, scatterplot

# %% Internal package import

from pyanno4rt.datahub import Datahub
from pyanno4rt.tools import (
    get_machine_learning_constraints, get_machine_learning_objectives)

# %% Set options

try:
    get_ipython().run_line_magic('matplotlib', 'qt5')
except AttributeError:
    pass

# %% Class definition


class MetricsGraphsPlotterMPL():
    """
    Data models metrics plot (matplotlib) class.

    This class provides metrics plots for the different data-dependent models.

    Attributes
    ----------
    DATA_DEPENDENT : bool
        Indicator for the assignment to model-related plots.

    category : string
        Plot category for assignment to the button groups in the visual \
        interface.

    name : string
        Attribute name of the classes' instance in the visual interface.

    label : string
        Label of the plot button in the visual interface.
    """

    # Set the flag for the model-related plots
    DATA_DEPENDENT = True

    # Set the class attributes for the visual interface integration
    category = "Data-driven model review"
    name = "metrics_graphs_plotter"
    label = "Evaluation metrics graphs"

    def view(self):
        """Open the full-screen view on the metrics plot."""
        # Initialize the datahub
        hub = Datahub()

        # Log a message about the plot opening
        hub.logger.display_info("Opening model evaluation metrics plot ...")

        def create_subtitle(figure, grid, title):
            """Create a subtitle for the plot row."""
            # Add the grid to the subplot
            row = figure.add_subplot(grid)

            # Set the title for the grid row
            row.set_title(f"{title}\n", fontweight='semibold', pad=20)

            # Hide the drawing of the rectangle patch
            row.set_frame_on(False)

            # Hide the axis
            row.axis('off')

        # Get the evaluation metrics data
        data = tuple((key,
                      value['auc_pr'],
                      value['auc_roc'],
                      value['f1'])
                     for key, value in hub.model_evaluations.items()
                     if any(component.model_parameters['model_label'] == key
                            for component in (
                                    get_machine_learning_constraints(
                                        hub.segmentation)
                                    + get_machine_learning_objectives(
                                        hub.segmentation))))

        # Unzip the data into the separate elements
        (evaluated_models, auc_prs, auc_rocs, f1s) = tuple(zip(*data))

        # Specify the evaluation modes
        modes = ('Training', 'Out-of-folds')

        # Get the number of graphs per model
        number_of_graphs = tuple(
            len(hub.model_instances[model_label]['display_options']['graphs'])
            for model_label in (*hub.model_instances,))

        # Loop over the number of evaluation subsets
        for i, _ in enumerate(evaluated_models):

            # Create a figure and subplots
            figure, axis = subplots(nrows=2, ncols=number_of_graphs[i],
                                    figsize=(8, 11), squeeze=False)

            # Initialize the row number and the grid layout object
            row_number = 0
            grid = GridSpec(2, number_of_graphs[i])

            # Loop over the number of modes
            for j, _ in enumerate(modes):

                # Initialize the column number
                column_number = 0

                # Check if the AUC-ROC scores should be displayed
                if 'AUC-ROC' in hub.model_instances[evaluated_models[i]][
                        'display_options']['graphs']:

                    # Plot the AUC-ROC points
                    scatterplot(
                        x="False Positive Rate", y="True Positive Rate",
                        data=auc_rocs[i][modes[j]]['curve'], s=50,
                        legend=False, ax=axis[row_number, column_number])

                    # Add the interpolation line to the AUC-ROC plot
                    axis[row_number, column_number].plot(
                        "False Positive Rate", "True Positive Rate",
                        data=auc_rocs[i][modes[j]]['curve'], lw=1, color='k')

                    # Add the diagonal line to the AUC-ROC plot
                    axis[row_number, column_number].plot(
                        linspace(0, 1, 100), linspace(0, 1, 100), color='k',
                        ls='--', lw=1)

                    # Fill the region under the interpolation line
                    axis[row_number, column_number].fill_between(
                        y1=auc_rocs[i][modes[j]]['curve']['True Positive Rate'],
                        x=auc_rocs[i][modes[j]]['curve']['False Positive Rate'],
                        alpha=.3, color='red')

                    # Set the plot title
                    axis[row_number, column_number].set_title(
                        "Receiver Operating Characteristic", fontsize=11)

                    # Add the AUC-ROC value to the plot
                    axis[row_number, column_number].annotate(
                        "".join((r"AUC$=$", str(round(
                            auc_rocs[i][modes[j]]['value'], 4)))),
                        xy=(0.82, 0.03), fontsize=8)

                    # Add the plot grid
                    axis[row_number, column_number].grid()

                    # Increment the column number
                    column_number += 1

                # Check if the AUC-PR scores should be displayed
                if 'AUC-PR' in hub.model_instances[evaluated_models[i]][
                        'display_options']['graphs']:

                    # Plot the AUC-PR line
                    lineplot(x="Recall", y="Precision",
                             data=auc_prs[i][modes[j]],
                             ax=axis[row_number, column_number])

                    # Set the limits for the y-axis
                    axis[row_number, column_number].set_ylim(0, 1)

                    # Set the plot title
                    axis[row_number, column_number].set_title(
                        "Precision-Recall Curve", fontsize=11)

                    # Add the plot grid
                    axis[row_number, column_number].grid()

                    # Increment the column number
                    column_number += 1

                # Check if the ROC scores should be displayed
                if 'F1' in hub.model_instances[evaluated_models[i]][
                        'display_options']['graphs']:

                    # Plot the F1 line
                    f1s[i][modes[j]]['values'].plot(
                        ax=axis[row_number, column_number],
                        ylim=(0, 1))

                    # Set the labels for x- and y-axis
                    axis[row_number, column_number].set_xlabel("Threshold")
                    axis[row_number, column_number].set_ylabel("F1 Score")

                    # Set the plot title
                    axis[row_number, column_number].set_title(
                        "F1 Curve", fontsize=11)

                    # Add a vertical line to indicate the best F1 position
                    axis[row_number, column_number].axvline(
                        f1s[i][modes[j]]['best'], lw=1, ls='--', color='k')

                    # Add the plot grid
                    axis[row_number, column_number].grid()

                # Create the subtitle for the plot row
                create_subtitle(
                    figure, grid[row_number, ::],
                    "".join((evaluated_models[i], " (", modes[j], ")")))

                # Increment the row number
                row_number += 1

            # Apply a tight layout to the figure
            figure.tight_layout()

            # Get the figure manager
            figure_manager = get_current_fig_manager()

            # Set the window title
            figure_manager.set_window_title("pyanno4rt - model evaluation "
                                            "graphs")

            # Show the plot in screen size
            figure_manager.window.showMaximized()
