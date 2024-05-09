"""Dose-volume histogram plot (matplotlib)."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from IPython import get_ipython
from itertools import islice, cycle
from matplotlib.pyplot import get_cmap, get_current_fig_manager, subplots
from numpy import ceil, linspace

# %% Internal package import

from pyanno4rt.datahub import Datahub

# %% Set options

try:
    get_ipython().run_line_magic('matplotlib', 'qt5')
except AttributeError:
    pass

# %% Class definition


class DVHGraphPlotterMPL():
    """
    Dose-volume histogram plot (matplotlib) class.

    This class provides a plot with dose-volume histogram curve per segment.

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
    name = "dvh_plotter"
    label = "Dose-volume histogram"

    def view(self):
        """Open the full-screen view on the dose-volume histogram."""
        # Initialize the datahub
        hub = Datahub()

        # Log a message about the plot opening
        hub.logger.display_info("Opening dose-volume histogram (DVH) ...")

        # Get the segmentation and histogram dictionaries from the datahub
        dose_histogram = hub.dose_histogram
        segments = tuple(segment for segment in (*dose_histogram,)
                         if segment != 'evaluation_points'
                         and segment in dose_histogram['display_segments'])

        # Get the colormap
        colors = get_cmap('turbo')(linspace(0, 1.0, len(segments)))

        # Set the line styles
        line_styles = tuple(
            islice(cycle(["-", "--", ":", "-."]), len(segments)))

        # Create a dictionary with the segment styles
        segment_styles = dict(
            zip(segments, tuple(zip(colors, line_styles))))

        # Create a figure and subplots
        figure, axis = subplots(figsize=(14, 8))

        # Add the dose-volume histogram curve for each segment
        for segment in segments:

            axis.plot(dose_histogram['evaluation_points'],
                      dose_histogram[segment]['dvh_values'],
                      linewidth=1.7,
                      color=segment_styles[segment][0],
                      linestyle=segment_styles[segment][1])

        # Set x- and y-label
        axis.set_xlabel("Dose per fraction [Gy]", fontsize=11)
        axis.set_ylabel("Relative volume [%]", fontsize=11)

        # Change the tick label sizes for both axes
        axis.tick_params(axis='both', which='major', labelsize=9)

        # Determine the step length on the x-axis
        x_step = min(
            (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100),
            key=lambda x: abs(ceil(
                max(dose_histogram['evaluation_points'])/x)-20))

        # Set x- and y-ticks
        axis.set_xticks(tuple(i*x_step for i in range(
            int(ceil(max(dose_histogram['evaluation_points']))/x_step)+1)))
        axis.set_yticks(tuple(i*5 for i in range(21)))

        # Set the x- and y-limits
        axis.set_xlim(left=-0.05)
        axis.set_ylim(-1, 101)

        # Set the facecolor for the axis
        axis.set_facecolor('whitesmoke')

        # Specify the grid with a subgrid
        axis.grid(which='major', color='lightgray', linewidth=0.8)
        axis.grid(which='minor', color='lightgray', linestyle=':',
                  linewidth=0.5)
        axis.minorticks_on()

        # Set the legend and its facecolor
        legend = axis.legend(segments, fontsize=9, framealpha=1)
        legend.get_frame().set_facecolor('snow')

        # Apply a tight layout to the figure
        figure.tight_layout()

        # Get the figure manager
        figure_manager = get_current_fig_manager()

        # Set the window title
        figure_manager.set_window_title("pyanno4rt - dose-volume histogram "
                                        "(DVH)")

        # Show the plot in screen size
        figure_manager.window.showMaximized()
