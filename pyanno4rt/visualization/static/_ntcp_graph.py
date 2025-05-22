"""(N)TCP graph."""

# Author: Tim Ortkamp

# %% External package import

from IPython import get_ipython
from matplotlib.pyplot import get_current_fig_manager, subplots
from numpy import ceil, divide

# %% Internal package import

from pyanno4rt.datahub import Datahub
from pyanno4rt.tools import (
    get_machine_learning_constraints, get_machine_learning_objectives,
    get_radiobiological_constraints, get_radiobiological_objectives)

# %% Set options

try:
    get_ipython().run_line_magic('matplotlib', 'qt5')
except AttributeError:
    pass

# %% Class definition


class NTCPGraph():
    """
    (N)TCP graph class.

    This class provides an (N)TCP graph for the data-driven models.
    """

    def view(self):
        """Open the (N)TCP graph."""

        # Initialize the datahub
        hub = Datahub()

        # Log a message about the plot opening
        hub.logger.display_info("Opening (N)TCP graph ...")

        # Get the segmentation data and the tracker
        segmentation = hub.segmentation
        tracker = hub.optimization['problem'].tracker

        # Determine the step length on the x-axis
        x_step = min(
            (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000,
             20000, 50000, 100000, 200000, 500000, 1000000),
            key=lambda x: abs(ceil(
                max(len(track) for track in tracker.values())/x)-20))

        # Get the legend labels, the tracks, and the model objectives
        # labels, tracks, model_objectives = get_labels_tracks_objectives()

        # Get the track labels and display flags
        track_ids, tracks, components = tuple(zip(*(
            (component.track_id,
             divide(tracker[component.track_id], component.weight),
             component) for component in (
                 get_machine_learning_constraints(segmentation)
                 + get_machine_learning_objectives(segmentation)
                 + get_radiobiological_constraints(segmentation)
                 + get_radiobiological_objectives(segmentation))
            if component.display)))

        # Create a figure and a subplot
        figure, axis = subplots(figsize=(14, 8))

        # Loop over the number of tracks
        for i, track in enumerate(tracks):

            # Plot the outcome values
            axis.plot(
                range(1, track.size+1), components[i].reverse(track), '.-')

        # Set x- and y-label
        axis.set_xlabel("Evaluation step", fontsize=11)
        axis.set_ylabel("(N)TCP", fontsize=11)

        # Change the tick label sizes for both axes
        axis.tick_params(axis='both', which='major', labelsize=9)

        # Set x- and y-ticks
        axis.set_xticks(
            tuple(i*x_step for i in range(int(ceil(max(
                len(track) for track in tracker.values())/x_step))+1)))
        axis.set_yticks(tuple(i/20 for i in range(21)))

        # Set the font sizes for the tick labels
        for label in axis.get_xticklabels() + axis.get_yticklabels():
            label.set_fontsize(9)

        # Set the x- and y-limits
        axis.set_xlim(0, max(len(track) for track in tracks)+1)
        axis.set_ylim(-0.01, 1.01)

        # Set the facecolor for the axis
        axis.set_facecolor("whitesmoke")

        # Specify the grid with a subgrid
        axis.grid(which='major', color='lightgray', linewidth=0.8)
        axis.grid(
            which='minor', color='lightgray', linestyle=':', linewidth=0.5)
        axis.minorticks_on()

        # Set the legend and its facecolor
        legend = axis.legend(track_ids, fontsize=9, framealpha=1)
        legend.get_frame().set_facecolor('snow')

        # Apply a tight layout
        figure.tight_layout()

        # Get the figure manager
        figure_manager = get_current_fig_manager()

        # Set the window title
        figure_manager.set_window_title("pyanno4rt - (N)TCP graph")

        # Show the full-screen plot
        figure_manager.window.showMaximized()
