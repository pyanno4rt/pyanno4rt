"""Dosimetrics table (matplotlib)."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from IPython import get_ipython
from matplotlib.pyplot import cm, get_current_fig_manager, subplots
from numpy import full
from pandas import DataFrame

# %% Internal package import

from pyanno4rt.datahub import Datahub

# %% Set options

try:
    get_ipython().run_line_magic('matplotlib', 'qt5')
except AttributeError:
    pass

# %% Class definition


class DosimetricsTablePlotterMPL():
    """
    Dosimetrics table (matplotlib) class.

    This class provides a table with dosimetric values per segment, e.g. mean \
    dose, dose deviation, min/max dose, DVH parameters and quality indicators.

    Attributes
    ----------
    category : string
        Plot category for assignment to the button groups in the visual \
        interface.

    name : string
        Attribute name of the classes' instance in the visual interface.

    label : string
        Label of the plot button in the visual interface.
    """

    # Set the class attributes for the visual interface integration
    category = "Treatment plan evaluation"
    name = "dosimetrics_plotter"
    label = "Dosimetric value table"

    def view(self):
        """Open the full-screen view on the dosimetrics table."""
        # Initialize the datahub
        hub = Datahub()

        # Log a message about the plot opening
        hub.logger.display_info("Opening matplotlib dosimetric value table "
                                "...")

        # Get the dosimetrics dictionary filtered by segment and metrics
        dosimetrics = {segment: {
            metric: hub.dosimetrics[segment][metric]
            for metric in hub.dosimetrics[segment]
            if any(display_metric in metric
                   for display_metric in hub.dosimetrics['display_metrics'])}
            for segment in hub.dosimetrics['display_segments']}

        # Convert the dosimetrics dictionary into a dataframe
        dataframe = DataFrame(dosimetrics).transpose().astype(float)

        # Create a figure and subplots
        figure, axis = subplots(figsize=(14, 8))

        # Set the axis off and tight
        axis.axis('off')
        axis.axis('tight')

        # Add the table to the axis
        table = axis.table(cellText=dataframe.values.round(4),
                           cellLoc='center',
                           rowLabels=dataframe.index,
                           colLabels=dataframe.columns,
                           rowColours=cm.BuPu(
                               full(dataframe.shape[0], 0.1)),
                           colColours=cm.BuPu(
                               full(dataframe.shape[1], 0.1)),
                           loc='center')

        # Disable auto-sized font and set font size manually
        table.auto_set_font_size(False)
        table.set_fontsize(8)

        # Apply a tight layout to the figure
        figure.tight_layout()

        # Get the figure manager
        figure_manager = get_current_fig_manager()

        # Set the window title
        figure_manager.set_window_title("pyanno4rt - dosimetric value table")

        # Show the plot in screen size
        figure_manager.window.showMaximized()
