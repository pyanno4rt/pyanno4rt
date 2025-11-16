"""DVH graph."""

# Author: Tim Ortkamp

# %% External package import

from itertools import islice, cycle
from matplotlib.pyplot import get_cmap, get_current_fig_manager, subplots
from numpy import ceil, linspace

# %% Internal package import

from pyanno4rt.tools import filter_dict

# %% Class definition


class DVHGraph():
    """
    DVH graph class.

    This class provides a plot with the segment-wise DVH values.

    Parameters
    ----------
    title : str, default=''
        Title of the plot.

    titlesize : int, default=16
        Font size of the title.

    xlabel : str, default='Dose [Gy]'
        Label for the x-axis.

    ylabel : str, default='Relative volume [%]',
        Label for the y-axis.

    labelsize : int, default=11
        Font size of the labels.

    linewidth : int, default=3
        Width of the line plots.

    ticksize : int, default=9
        Font size of the axis ticks.

    legendsize : int, default=9
        Font size of the legend.

    background : {'lightgray', 'white', 'whitesmoke'}, default='whitesmoke'
        Background color for the plot.

    gridlines : bool, default=True
        Indicator for the display of the gridlines.

    gridcolor : {'black', 'darkgray', 'lightgray'}, default='lightgray'
        Color of the gridlines.

    Attributes
    ----------
    title : str
        See 'Parameters'.

    titlesize : int
        See 'Parameters'.

    xlabel : str
        See 'Parameters'.

    ylabel : str
        See 'Parameters'.

    labelsize : int
        See 'Parameters'.

    linewidth : int
        See 'Parameters'.

    ticksize : int
        See 'Parameters'.

    legendsize : int
        See 'Parameters'.

    background : {'lightgray', 'white', 'whitesmoke'}
        See 'Parameters'.

    gridlines : bool
        See 'Parameters'.

    gridcolor : {'black', 'darkgray', 'lightgray'}
        See 'Parameters'.
    """

    def __init__(
            self,
            title='',
            titlesize=16,
            xlabel='Dose [Gy]',
            ylabel='Relative volume [%]',
            labelsize=11,
            linewidth=3,
            ticksize=9,
            legendsize=9,
            background='whitesmoke',
            gridlines=True,
            gridcolor='lightgray'):

        # Get the input arguments
        inputs = filter_dict(vars(), remove_keys=('self',))

        # Loop over the input arguments
        for key, value in inputs.items():

            # Set the attribute
            setattr(self, key, value)

    def view(
            self,
            treatment_plan,
            identifiers):
        """
        Open the DVH graph.

        Parameters
        ----------
        treatment_plan : object of class \
            :class:`~pyanno4rt.base._treatment_plan.TreatmentPlan`
            The object used to represent the treatment plan.

        identifiers : None or list
            Segment identifiers for filtering.
        """

        # Set the value for the segment identifiers
        identifiers = [] if identifiers is None else identifiers

        # Get the DVH data
        dose_histogram = treatment_plan.datahub.dose_histogram

        # Get the segment names
        segments = tuple(
            segment for segment in dose_histogram
            if segment != 'evaluation_points')

        # Set the colormap
        colors = get_cmap('tab20b')(linspace(0, 1.0, len(segments)))

        # Set the line styles
        lines = tuple(islice(cycle(["-", "--", ":", "-."]), len(segments)))

        # Create a dictionary for the track styles
        styles = dict(
            zip(segments, tuple(zip(colors, lines))))

        # Set the expected number of ticks
        number_of_ticks = 20 if self.ticksize < 17 else 10

        # Check if segment identifiers have been passed
        if len(identifiers) > 0:

            # Reduce the segment names
            segments = tuple(
                segment for segment in segments if segment in identifiers)

        # Get the figure and axis objects
        figure, axis = subplots(figsize=(14, 8))

        # Set the plot title
        axis.set_title(
            label=self.title, fontsize=self.titlesize, fontweight='semibold',
            pad=10)

        # Loop over the segments
        for segment in segments:

            # Plot the DVH curve
            axis.plot(
                dose_histogram['evaluation_points'],
                100*dose_histogram[segment]['dvh_values'],
                color=styles[segment][0],
                linestyle=styles[segment][1],
                linewidth=0.5*self.linewidth)

        # Set the x- and y-labels
        axis.set_xlabel(xlabel=self.xlabel, fontsize=self.labelsize)
        axis.set_ylabel(ylabel=self.ylabel, fontsize=self.labelsize)

        # Configure the axis ticks
        axis.tick_params(axis='both', which='major', labelsize=self.ticksize)

        # Determine the step length on the x-axis
        x_step = min(
            sorted(base*10**i for base in (1, 2, 5) for i in range(-6, 6)),
            key=lambda x: abs(ceil(max(dose_histogram['evaluation_points'])/x)
                              - number_of_ticks))

        # Set the x- and y-ticks
        axis.set_xticks(tuple(i*x_step for i in range(
            int(ceil(max(dose_histogram['evaluation_points']))/x_step)+1)))
        axis.set_yticks(tuple(i*5 for i in range(number_of_ticks+1)))

        # Set the x- and y-limits
        axis.set_xlim(left=-0.05)
        axis.set_ylim(-1, 101)

        # Set the facecolor for the axis
        axis.set_facecolor(self.background)

        # Specify the grid with a subgrid
        axis.grid(which='major', color=self.gridcolor, linewidth=0.8)
        axis.grid(
            which='minor', color=self.gridcolor, linestyle=':', linewidth=0.5)

        # Check if the grid should be displayed
        if self.gridlines:

            # Enable the grids
            axis.grid(True)
            axis.minorticks_on()

        else:

            # Disable the grids
            axis.grid(False)
            axis.minorticks_off()

        # Configure the legend
        legend = axis.legend(segments, fontsize=self.legendsize, framealpha=1)
        legend.get_frame().set_facecolor('snow')

        # Apply a tight layout
        figure.tight_layout()

        # Get the figure manager
        figure_manager = get_current_fig_manager()

        # Set the window title
        figure_manager.set_window_title("DVH graph")

        # Show the full-screen plot
        figure_manager.window.showMaximized()
