"""Dosimetrics table."""

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


class DosimetricsTable():
    """
    Dosimetrics table class.

    This class provides a dosimetrics table, including segment-wise dose \
    statistics, DVH parameters and quality indicators.
    """

    def view(self, plan):
        """Open the dosimetrics table."""

        # Convert the dosimetrics dictionary into a dataframe
        dataframe = DataFrame(
            plan.dosimetrics.quantities).transpose().astype(float)

        # Create a figure and subplots
        figure, axis = subplots(figsize=(14, 8))

        # Set the axis off and tight
        axis.axis('off')
        axis.axis('tight')

        # Add the table to the axis
        table = axis.table(
            cellText=dataframe.values.round(4),
            cellLoc='center',
            rowLabels=dataframe.index,
            colLabels=dataframe.columns,
            rowColours=cm.BuPu(full(dataframe.shape[0], 0.1)),
            colColours=cm.BuPu(full(dataframe.shape[1], 0.1)),
            loc='center')

        # Disable auto-sized font and set font size manually
        table.auto_set_font_size(False)
        table.set_fontsize(8)

        # Apply a tight layout
        figure.tight_layout()

        # Get the figure manager
        figure_manager = get_current_fig_manager()

        # Set the window title
        figure_manager.set_window_title("pyanno4rt - dosimetrics table")

        # Show the full-screen plot
        figure_manager.window.showMaximized()
