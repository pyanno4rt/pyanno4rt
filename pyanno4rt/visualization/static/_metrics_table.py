"""Metrics table."""

# Author: Tim Ortkamp

# %% External package import

from IPython import get_ipython
from matplotlib.pyplot import cm, get_current_fig_manager, subplots
from numpy import full
from pandas import DataFrame

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

    def view(
            self,
            treatment_plan,
            model_name):
        """Open the metrics table."""

        def dict_to_dataframe(indicators):
            """Convert the indicator dictionary into a dataframe."""

            # Get the dataframe
            dataframe = DataFrame(indicators).transpose().astype(float)

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

            # Modify the column names
            dataframe.columns = [
                r'{}{}'.format(el[0], el[1]) for el in zip(
                    dataframe.columns,
                    tuple(arrows[column] for column in dataframe.columns))]

            return (
                dataframe.values.round(4), dataframe.index, dataframe.columns)

        # Get the model
        model = next((
            model for model in treatment_plan.data_model_handler.models
            if model.label == model_name), None)

        # Get the KPI data
        indicators = model.evaluator.results['kpi']

        # Convert the indicator dictionary into a dataframe
        dataframe = dict_to_dataframe(indicators)

        # Create a figure and subplots
        figure, axis = subplots(
            nrows=1, ncols=1, figsize=(14, 8), squeeze=False)

        # Set the figure patch to invisible
        figure.patch.set_visible(False)

        # Disable the plot axis
        axis[0, 0].axis('off')

        # Specify the row and column header colors
        row_colors = cm.BuPu(full(len(dataframe[1]), 0.1))
        column_colors = cm.BuPu(full(len((*dataframe[2],)), 0.1))

        # Set the title for the table
        axis[0, 0].set_title(model.label, fontweight='semibold', pad=10)

        # Generate the table from the dataframe
        table = axis[0, 0].table(
            cellText=dataframe[0],
            cellLoc='center',
            rowLabels=dataframe[1],
            colLabels=dataframe[2],
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
        figure_manager.set_window_title("Metrics Table")

        # Show the full-screen plot
        figure_manager.window.showMaximized()
