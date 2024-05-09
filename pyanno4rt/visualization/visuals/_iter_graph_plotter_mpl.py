"""Iterative objective value plot (matplotlib)."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from IPython import get_ipython
from matplotlib.pyplot import get_current_fig_manager, subplots
from numpy import ceil, floor

# %% Internal package import

from pyanno4rt.datahub import Datahub
from pyanno4rt.tools import (
    get_all_constraints, get_all_objectives, get_constraint_segments,
    get_objective_segments)

# %% Set options

try:
    get_ipython().run_line_magic('matplotlib', 'qt5')
except AttributeError:
    pass

# %% Class definition


class IterGraphPlotterMPL():
    """
    Iterative objective value plot (Matplotlib) class.

    This class provides a plot with the iterative objective function values.

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
    category = "Optimization problem analysis"
    name = "iterations_plotter"
    label = "Iterative objective value plot"

    def view(self):
        """Open the full-screen view on the iterative objective value plot."""
        # Initialize the datahub
        hub = Datahub()

        # Log a message about the plot opening
        hub.logger.display_info("Opening iterative objective value plot ...")

        def get_plotting_information():
            """Get the labels for the plot legend."""

            # Determine the segment/component groups
            groups = (tuple(zip(
                get_objective_segments(segmentation),
                get_all_objectives(segmentation)))
                + tuple(zip(
                    get_constraint_segments(segmentation),
                    get_all_constraints(segmentation))))

            # Convert the groups into an appropriate format
            groups = ((group[0], group[1].name) if group[1].link is None
                      else ("[{}]".format(", ".join([group[0]]+group[1].link)),
                            group[1].name)
                      for group in groups)

            # Get the display flags from the objectives
            display_flags = [component.display for component in (
                        get_all_objectives(segmentation)
                        + get_all_constraints(segmentation))]

            # Build the legend labels
            legend_labels = [r" $\rightarrow$ ".join(
                (group[0], group[1])) for index, group in enumerate(groups)
                if display_flags[index]]

            return legend_labels, display_flags

        # Get the segmentation data from the datahub
        segmentation = hub.segmentation

        # Get the legend labels and display flags
        legend_labels, display_flags = get_plotting_information()

        # Get the tracks from the datahub which should be displayed
        tracker = {track[0]: track[1] for index, track in enumerate(
            hub.optimization['problem'].tracker.items())
                   if display_flags[index]}

        # Determine the step length on the x-axis
        x_step = min(
            (1, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000),
            key=lambda x: abs(ceil(
                max(len(track) for track in tracker.values())/x)-20))

        # Determine the step length on the y-axis
        y_step = min(
             (5e-5, 5e-4, 5e-3, 5e-2, 5e-1, 5e0, 5e1, 5e2, 5e3, 5e4, 5e5),
             key=lambda x: abs(ceil((
                 max(max(track) for track in tracker.values())
                 - min(min(track) for track in tracker.values()))/x)-20))

        # Create a figure and subplots
        figure, axis = subplots(figsize=(14, 8))

        # Loop over the recorded tracks
        for track in tracker.values():

            # Plot the track
            axis.plot(range(1, len(track)+1), track, '.-')

        # Set x- and y-label
        axis.set_xlabel("Evaluation step", fontsize=11)
        axis.set_ylabel("Objective value", fontsize=11)

        # Change the tick label sizes for both axes
        axis.tick_params(axis='both', which='major', labelsize=9)

        # Set x- and y-ticks
        axis.set_xticks(tuple(i*x_step for i in range(
            int(ceil(max(len(track) for track in tracker.values())/x_step))+1))
            )
        axis.set_yticks(tuple(i*y_step for i in range(
            int(floor(min(min(track) for track in tracker.values())/y_step))-1,
            int(ceil(max(max(track) for track in tracker.values())/y_step))+1))
            )

        # Set the font sizes for the tick labels
        for label in axis.get_xticklabels():
            label.set_fontsize(9)
        for label in axis.get_yticklabels():
            label.set_fontsize(9)

        # Set the x- and y-limits
        axis.set_xlim(0, max(len(track) for track in tracker.values())+x_step)
        axis.set_ylim(
            min(min(track) for track in tracker.values()) - y_step,
            max(max(track) for track in tracker.values()) + y_step)

        # Set the facecolor for the axis
        axis.set_facecolor("whitesmoke")

        # Specify the grid with a subgrid
        axis.grid(which='major', color='lightgray', linewidth=0.8)
        axis.grid(which='minor', color='lightgray', linestyle=':',
                  linewidth=0.5)
        axis.minorticks_on()

        # Set the legend and its facecolor
        legend = axis.legend(legend_labels, fontsize=9, framealpha=1)
        legend.get_frame().set_facecolor('snow')

        # Apply a tight layout to the figure
        figure.tight_layout()

        # Get the figure manager
        figure_manager = get_current_fig_manager()

        # Set the window title
        figure_manager.set_window_title("pyanno4rt - iterative objective "
                                        "value plot")

        # Show the plot in screen size
        figure_manager.window.showMaximized()
