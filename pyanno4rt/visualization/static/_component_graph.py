"""Iterative component graph."""

# Author: Tim Ortkamp

# %% External package import

from itertools import islice, cycle
from matplotlib.pyplot import get_cmap, get_current_fig_manager, subplots
from numpy import ceil, floor, linspace

# %% Internal package import

from pyanno4rt.tools import (
    filter_dict, flatten, get_all_constraints, get_all_objectives)

# %% Class definition


class ComponentGraph():
    """
    Iterative component graph class.

    This class provides a plot with the iteration-wise values of the \
    optimization functions.

    Parameters
    ----------
    title : str, default=''
        Title of the plot.

    titlesize : int, default=16
        Font size of the title.

    xlabel : str, default='Evaluation step'
        Label for the x-axis.

    ylabel : str, default='Component value',
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
            xlabel='Evaluation step',
            ylabel='Component value',
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
            identifiers=None):
        """
        Open the iterative component graph.

        Parameters
        ----------
        treatment_plan : object of class \
            :class:`~pyanno4rt.base._treatment_plan.TreatmentPlan`
            The object used to represent the treatment plan.

        ids : None or list
            Track identifiers for filtering.
        """

        # Set the value for the track identifiers
        identifiers = [] if identifiers is None else identifiers

        # Get the segmentation and optimization data
        segmentation, optimization = (
            getattr(treatment_plan.datahub, attribute) for attribute in (
                'segmentation', 'optimization'))

        # Get all optimization components
        components = (
            get_all_objectives(segmentation)
            + get_all_constraints(segmentation))

        # Get the tracks to be displayed
        tracker = {
            component.track_id: (
                optimization['problem'].tracker[component.track_id])
            for component in components if component.display}

        # Get the track statistics
        track_min = min(flatten(tracker.values()))
        track_max = max(flatten(tracker.values()))
        track_len = max(len(track) for track in tracker.values())
        track_num = len(tracker)

        # Set the expected number of ticks
        number_of_ticks = 20 if self.ticksize < 17 else 10

        # Determine the step length on the x-axis
        x_step = min(
            sorted(base*10**i for base in (1, 2, 5) for i in range(6)),
            key=lambda x: abs(ceil(track_len/x)-number_of_ticks))

        # Determine the step length on the y-axis
        y_step = min(
            sorted(base*10**i for base in (1, 2, 5) for i in range(-6, 6)),
            key=lambda x: abs(ceil((track_max-track_min)/x)-number_of_ticks))

        # Set the marker styles
        markers = tuple(
            islice(cycle(['o', 's', 'v', 'd', '*', 'X']), track_num))

        # Set the colormap
        colors = get_cmap('tab20b')(linspace(0, 1.0, track_num))

        # Set the line styles
        lines = tuple(islice(cycle(["-", "--", ":", "-."]), track_num))

        # Create a dictionary for the track styles
        styles = dict(
            zip(tracker, tuple(zip(markers, colors, lines))))

        # Check if track identifiers have been passed
        if len(identifiers) > 0:

            # Reduce the tracker
            tracker = {
                key: value for key, value in tracker.items()
                if key in identifiers}

        # Get the figure and axis objects
        figure, axis = subplots(figsize=(14, 8))

        # Set the plot title
        axis.set_title(
            label=self.title, fontsize=self.titlesize, fontweight='semibold',
            pad=10)

        # Loop over the tracks
        for track, values in tracker.items():

            # Plot the track
            axis.plot(
                range(1, len(values)+1),
                values,
                marker=styles[track][0],
                markersize=1.5*self.linewidth,
                color=styles[track][1],
                linestyle=styles[track][2],
                linewidth=0.5*self.linewidth)

        # Set the x- and y-labels
        axis.set_xlabel(xlabel=self.xlabel, fontsize=self.labelsize)
        axis.set_ylabel(ylabel=self.ylabel, fontsize=self.labelsize)

        # Configure the axis ticks
        axis.tick_params(axis='both', which='major', labelsize=self.ticksize)

        # Set the x- and y-ticks
        axis.set_xticks(tuple(
            i*x_step for i in range(int(ceil(track_len/x_step))+1)))
        axis.set_yticks(tuple(
            i*y_step for i in range(
                int(floor(track_min/y_step))-1,
                int(ceil(track_max/y_step))+1)))

        # Set the x- and y-limits
        axis.set_xlim(0, track_len+x_step/2)
        axis.set_ylim(track_min-y_step/2, track_max+y_step/2)

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
        legend = axis.legend(tracker, fontsize=self.legendsize, framealpha=1)
        legend.get_frame().set_facecolor('snow')

        # Apply a tight layout
        figure.tight_layout()

        # Get the figure manager
        figure_manager = get_current_fig_manager()

        # Set the window title
        figure_manager.set_window_title("Iterative component graph")

        # Show the full-screen plot
        figure_manager.window.showMaximized()
