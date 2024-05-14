"""Data models metrics table (matplotlib)."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from IPython import get_ipython
from matplotlib.pyplot import cm, get_current_fig_manager, subplots
from numpy import full
from pandas import DataFrame

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


class MetricsTablesPlotterMPL():
    """
    Data models metrics table (matplotlib) class.

    This class provides the metrics table for the different data-dependent \
    models.

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
    name = "metrics_tables_plotter"
    label = "Evaluation metrics tables"

    def view(self):
        """Open the full-screen view on the metrics table."""
        # Initialize the datahub
        hub = Datahub()

        # Log a message about the plot opening
        hub.logger.display_info("Opening model evaluation tables ...")

        def dict_to_dataframe(indicators, display_metrics):
            """Convert the indicator dictionaries into dataframe elements."""
            # Get the dataframes
            dataframes = tuple(
                DataFrame(subdict).transpose().astype(float)[
                    display_metrics[index]]
                for index, subdict in enumerate(indicators))

            # Map the up- and downarrows to the metrics
            arrows = {'Logloss': r'$\downarrow$',
                      'Brier score': r'$\downarrow$',
                      'Subset accuracy': r'$\uparrow$',
                      'Cohen Kappa': r'$\uparrow$',
                      'Hamming loss': r'$\downarrow$',
                      'Jaccard score': r'$\uparrow$',
                      'Precision': r'$\uparrow$',
                      'Recall': r'$\uparrow$',
                      'F1 score': r'$\uparrow$',
                      'MCC': r'$\uparrow$',
                      'AUC': r'$\uparrow$'}

            # Modify each dataframe column
            for dataframe in dataframes:
                dataframe.columns = [
                    r'{}{}'.format(el[0], el[1]) for el in zip(
                        dataframe.columns,
                        tuple(arrows[column] for column in dataframe.columns))]

            return tuple((dataframe.values.round(4),
                          dataframe.index,
                          dataframe.columns)
                         for dataframe in dataframes)

        # Get the evaluation data
        data = tuple((key, value['kpi'],
                      hub.model_instances[key][
                          'display_options']['kpis'])
                     for key, value in hub.model_evaluations.items()
                     if any(component.model_parameters['model_label'] == key
                            for component in (
                                    get_machine_learning_constraints(
                                        hub.segmentation)
                                    + get_machine_learning_objectives(
                                        hub.segmentation))))

        # Unzip the data into separate elements
        model_names, indicators, display_metrics = tuple(zip(*data))

        # Convert the indicator dictionaries into dataframes
        dataframes = dict_to_dataframe(indicators, display_metrics)

        # Create a figure and subplots
        figure, axis = subplots(nrows=len(dataframes), ncols=1,
                                figsize=(14, 8), squeeze=False)

        # Set the figure patch to invisible
        figure.patch.set_visible(False)

        # Loop over the number of dataframes
        for i, _ in enumerate(dataframes):

            # Disable the plot axis
            axis[i, 0].axis('off')

            # Specify the row and column header colors
            row_colors = cm.BuPu(full(len(dataframes[i][1]), 0.1))
            column_colors = cm.BuPu(full(len((*dataframes[i][2],)), 0.1))

            # Set the title for the table
            axis[i, 0].set_title(model_names[i], fontweight='semibold')

            # Generate the table from the dataframe
            table = axis[i, 0].table(cellText=dataframes[i][0],
                                     cellLoc='center',
                                     rowLabels=dataframes[i][1],
                                     colLabels=dataframes[i][2],
                                     rowColours=row_colors,
                                     colColours=column_colors,
                                     loc='center')

            # Set the font size manually
            table.auto_set_font_size(False)
            table.set_fontsize(8)

        # Apply a tight layout to the figure
        figure.tight_layout()

        # Get the figure manager
        figure_manager = get_current_fig_manager()

        # Set the window title
        figure_manager.set_window_title("pyanno4rt - model evaluation tables")

        # Show the plot in screen size
        figure_manager.window.showMaximized()
