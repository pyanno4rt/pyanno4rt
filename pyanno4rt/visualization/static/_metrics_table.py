"""Metrics table."""

# Author: Tim Ortkamp

# %% External package import

from IPython import get_ipython
from matplotlib.pyplot import cm, get_current_fig_manager, subplots
from numpy import full
from pandas import DataFrame

# %% Internal package import

from pyanno4rt.tools import (
    get_machine_learning_constraints, get_machine_learning_objectives)

# %% Set options

try:
    get_ipython().run_line_magic('matplotlib', 'qt5')
except AttributeError:
    pass

# %% Class definition


class MetricsTable():
    """
    Metrics table class.

    This class provides a metrics table for the data-driven models.
    """

    def view(self):
        """Open the metrics table."""

        # Initialize the datahub
        hub = Datahub()

        def dict_to_dataframe(indicators, display_metrics):
            """Convert the indicator dictionaries into dataframe elements."""

            # Get the dataframes
            dataframes = tuple(
                DataFrame(subdict).transpose().astype(float)[
                    display_metrics[index]]
                for index, subdict in enumerate(indicators))

            # Map the up- and downarrows to the metrics
            arrows = {
                'Logloss': r'$\downarrow$',
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

            # Loop over the dataframes
            for dataframe in dataframes:

                # Modify the column names
                dataframe.columns = [
                    r'{}{}'.format(el[0], el[1]) for el in zip(
                        dataframe.columns,
                        tuple(arrows[column] for column in dataframe.columns))]

            return tuple((
                dataframe.values.round(4), dataframe.index, dataframe.columns)
                for dataframe in dataframes)

        # Get the evaluation data
        data = tuple((
            key, value['kpi'],
            hub.model_instances[key]['display_options'].kpis)
            for key, value in hub.model_evaluations.items()
            if any(component.model_parameters.model_label == key
                   for component in (
                           get_machine_learning_constraints(hub.segmentation)
                           + get_machine_learning_objectives(hub.segmentation)
                           )))

        # Unzip the data into separate elements
        model_names, indicators, display_metrics = tuple(zip(*data))

        # Convert the indicator dictionaries into dataframes
        dataframes = dict_to_dataframe(indicators, display_metrics)

        # Create a figure and subplots
        figure, axis = subplots(
            nrows=len(dataframes), ncols=1, figsize=(14, 8), squeeze=False)

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
            axis[i, 0].set_title(model_names[i], fontweight='semibold', pad=10)

            # Generate the table from the dataframe
            table = axis[i, 0].table(
                cellText=dataframes[i][0],
                cellLoc='center',
                rowLabels=dataframes[i][1],
                colLabels=dataframes[i][2],
                rowColours=row_colors,
                colColours=column_colors,
                loc='center')

            # Set the font size manually
            table.auto_set_font_size(False)
            table.set_fontsize(8)

        # Apply a tight layout
        figure.tight_layout()

        # Get the figure manager
        figure_manager = get_current_fig_manager()

        # Set the window title
        figure_manager.set_window_title("pyanno4rt - metrics table")

        # Show the full-screen plot
        figure_manager.window.showMaximized()
