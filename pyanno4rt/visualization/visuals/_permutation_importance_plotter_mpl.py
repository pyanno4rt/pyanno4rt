"""Data models permutation importance plot (matplotlib)."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from IPython import get_ipython
from matplotlib.pyplot import get_current_fig_manager, subplots
from numpy import repeat
from pandas import DataFrame, melt
from seaborn import boxplot as sns_boxplot
from seaborn import color_palette

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


class PermutationImportancePlotterMPL():
    """
    Data models permutation importance plot (matplotlib) class.

    This class provides permutation importance plots for the different \
    data-dependent models.

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
    name = "permutation_importance_plotter"
    label = "Permutation importance boxplots"

    def view(self):
        """Open the full-screen view on the permutation importance plot."""
        # Initialize the datahub
        hub = Datahub()

        # Log a message about the plot opening
        hub.logger.display_info("Opening model permutation importance plot "
                                "...")

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

        # Get the inspection data
        data = tuple((key,
                      DataFrame(
                          data=value['permutation_importance']['Training'],
                          columns=hub.datasets[key]['feature_names']),
                      DataFrame(
                          data=value['permutation_importance']['Out-of-folds'],
                          columns=hub.datasets[key]['feature_names']))
                     for key, value in hub.model_inspections.items()
                     if any(component.model_parameters['model_label'] == key
                            for component in (
                                    get_machine_learning_constraints(
                                        hub.segmentation)
                                    + get_machine_learning_objectives(
                                        hub.segmentation))))

        # Unzip the data into the separate elements
        data_zipped = list(zip(*data))

        # Get the number of features to display
        number_to_display = tuple(
            min((10, len(value['feature_names'])))
            for key, value in hub.datasets.items())

        # Preprocess the training permutation importances
        data_zipped[1] = tuple(dataframe.reindex(
            dataframe.mean().sort_values(ascending=False).index,
            axis=1).iloc[:, :number_to_display[i]]
            for i, dataframe in enumerate(data_zipped[1]))

        # Get the number of permutation repetitions from the training set
        number_of_repeats = tuple(len(dataframe)
                                  for dataframe in data_zipped[1])

        # Preprocess the validation permutation importances
        data_zipped[2] = tuple(
            melt(dataframe.reindex(
                dataframe.mean().sort_values(ascending=False).index,
                axis=1).iloc[:, :number_to_display[i]].assign(
                    fold=repeat([1, 2, 3, 4, 5], number_of_repeats[i])),
                'fold', var_name='feature', value_name='importance')
            for i, dataframe in enumerate(data_zipped[2]))

        # Loop over the number of inspected models
        for i, _ in enumerate(data_zipped[0]):

            # Create a figure and subplots
            figure, axis = subplots(nrows=2, ncols=1, figsize=(14, 8))

            # Plot the training permutation importance boxplots
            sns_boxplot(data=data_zipped[1][i], ax=axis[0])

            # Plot the validation permutation importance boxplots
            sns_boxplot(data=data_zipped[2][i], x='feature', y='importance',
                        hue='fold', palette=color_palette(), ax=axis[1])

            for j in range(0, 2):

                # Set x- and y-label
                axis[j].set_xlabel("Feature", fontsize=11)
                axis[j].set_ylabel("Importance value", fontsize=11)

                # Change the tick label sizes for both axes
                axis[j].tick_params(axis='both', which='major', labelsize=9)

                # Rotate the tick labels
                axis[j].tick_params(axis='x', labelrotation=15)

                # Set the facecolor for the axis
                axis[j].set_facecolor("whitesmoke")

                # Specify the grid with a subgrid
                axis[j].grid(which='major', color='lightgray', linewidth=0.8)
                axis[j].grid(which='minor', color='lightgray', linestyle=':',
                             linewidth=0.5)
                axis[j].minorticks_on()

                # Hide the axis behind the boxplots
                axis[j].set_axisbelow(True)

            # Create the subtitles for the plot rows
            for k, subset in enumerate(('Training', 'Out-of-folds')):
                axis[k].set_title("".join((data_zipped[0][i], " (",
                                           subset, ")", ": top-{} features"
                                           .format(number_to_display[i]))),
                                  fontweight='semibold', pad=20)

            # Apply a tight layout to the figure
            figure.tight_layout()

            # Get the figure manager
            figure_manager = get_current_fig_manager()

            # Set the window title
            figure_manager.set_window_title("pyanno4rt - permutation "
                                            "importance boxplots")

            # Show the plot in screen size
            figure_manager.window.showMaximized()
