"""Permutation importance boxplot."""

# Author: Tim Ortkamp

# %% External package import

from IPython import get_ipython
from matplotlib.pyplot import get_current_fig_manager, subplots
from pandas import DataFrame
from seaborn import boxplot as sns_boxplot

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


class PermutationImportanceBoxplot():
    """
    Permutation importance boxplot class.

    This class provides a permutation importance boxplot for the data-driven \
    models.
    """

    def view(self):
        """Open the permutation importance boxplot."""

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

        # Initialize the datahub
        hub = Datahub()

        # Log a message about the plot opening
        hub.logger.display_info("Opening permutation importance boxplot ...")

        # Get the inspection data
        data = tuple((
            key,
            value['permutation_importance']['score'],
            DataFrame(
                data=value['permutation_importance']['Training'],
                columns=hub.datasets[key]['feature_names']),
            DataFrame(
                data=value['permutation_importance']['Out-of-folds'],
                columns=hub.datasets[key]['feature_names']))
            for key, value in hub.model_inspections.items()
            if any(component.model_parameters.model_label == key
                   for component in (
                           get_machine_learning_constraints(hub.segmentation)
                           + get_machine_learning_objectives(hub.segmentation))
                   ))

        # Unzip the data into the separate elements
        data_zipped = list(zip(*data))

        # Get the number of features to display
        number_to_display = tuple(
            min((10, len(value['feature_names'])))
            for key, value in hub.datasets.items())

        # Preprocess the training permutation importances
        data_zipped[2] = tuple(dataframe.reindex(
            dataframe.mean().sort_values(ascending=False).index,
            axis=1).iloc[:, :number_to_display[i]]
            for i, dataframe in enumerate(data_zipped[2]))

        # Preprocess the out-of-folds permutation importances
        data_zipped[3] = tuple(dataframe.reindex(
            dataframe.mean().sort_values(ascending=False).index,
            axis=1).iloc[:, :number_to_display[i]]
            for i, dataframe in enumerate(data_zipped[3]))

        # Loop over the number of inspected models
        for i, _ in enumerate(data_zipped[0]):

            # Create a figure and subplots
            figure, axis = subplots(nrows=2, ncols=1, figsize=(14, 8))

            # Plot the training permutation importance boxplot
            sns_boxplot(data=data_zipped[2][i], ax=axis[0])

            # Plot the out-of-folds permutation importance boxplot
            sns_boxplot(data=data_zipped[3][i], ax=axis[1])

            # Loop over the training and out-of-folds subsets
            for j, subset in enumerate(('Training', 'Out-of-folds')):

                # Set x- and y-label
                axis[j].set_xlabel("Feature", fontsize=11)
                axis[j].set_ylabel(f"Δ {data_zipped[1][j]}", fontsize=11)

                # Set the axis title
                axis[j].set_title(
                    f'{data_zipped[0][i]} ({subset}): '
                    f'top-{number_to_display[i]} features',
                    fontweight='semibold', pad=20)

                # Change the tick label sizes for both axes
                axis[j].tick_params(axis='both', which='major', labelsize=9)

                # Rotate the tick labels
                axis[j].tick_params(axis='x', labelrotation=15)

                # Set the facecolor
                axis[j].set_facecolor("whitesmoke")

                # Specify the grid properties
                axis[j].grid(which='major', color='lightgray', linewidth=0.8)
                axis[j].grid(
                    which='minor', color='lightgray', linestyle=':',
                    linewidth=0.5)
                axis[j].minorticks_on()

                # Hide the axis behind the boxplots
                axis[j].set_axisbelow(True)

            # Apply a tight layout
            figure.tight_layout()

            # Get the figure manager
            figure_manager = get_current_fig_manager()

            # Set the window title
            figure_manager.set_window_title(
                "pyanno4rt - permutation importance boxplot")

            # Show the full-screen plot
            figure_manager.window.showMaximized()
