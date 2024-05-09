"""Iterative (N)TCP value plot (matplotlib)."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from IPython import get_ipython
from itertools import tee
from matplotlib.pyplot import get_current_fig_manager, subplots
from numpy import array, ceil, divide

# %% Internal package import

from pyanno4rt.datahub import Datahub
from pyanno4rt.tools import (
    get_all_constraints, get_all_objectives, get_constraint_segments,
    get_objective_segments, sigmoid)

# %% Set options

try:
    get_ipython().run_line_magic('matplotlib', 'qt5')
except AttributeError:
    pass

# %% Class definition


class NTCPGraphPlotterMPL():
    """
    Iterative (N)TCP value plot (matplotlib) class.

    This class provides a plot with the iterative (N)TCP values from each \
    outcome prediction model.

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

    # Set the flag for the model-related plots
    DATA_DEPENDENT = True

    # Set the class attributes for the visual interface integration
    category = "Optimization problem analysis"
    name = "ntcp_plotter"
    label = "Iterative (N)TCP value plot"

    def view(self):
        """Open the full-screen view on the iterative (N)TCP value plot."""
        # Initialize the datahub
        hub = Datahub()

        # Log a message about the plot opening
        hub.logger.display_info("Opening iterative (N)TCP value plot ...")

        # Get the segmentation data and the tracker from the datahub
        segmentation = hub.segmentation
        tracker = hub.optimization['problem'].tracker

        def get_labels_tracks_objectives():
            """Get the legend labels, the tracks, and the model objectives."""

            # Determine the segment/objective pairs
            groups_obj = tuple(group for group in tuple(zip(
                get_objective_segments(segmentation),
                get_all_objectives(segmentation)))
                if group[1].RETURNS_OUTCOME and group[1].display)

            # Convert the pairs into an appropriate format
            groups_obj = ((group[0], str([group[0]]), group[1].name,
                           group[1].weight, group[1])
                          if group[1].link is None
                          else ("[{}]".format(", ".join([group[0]]+group[1].link)),
                                str([group[0]]+group[1].link), group[1].name,
                                group[1].weight, group[1])
                          for group in groups_obj)

            # Multiplicate the pairs generator
            groups_obj = tee(groups_obj, 3)

            # Determine the segment/constraint pairs
            groups_cons = tuple(group for group in tuple(zip(
                get_constraint_segments(segmentation),
                get_all_constraints(segmentation)))
                if group[1].RETURNS_OUTCOME and group[1].display)

            # Convert the pairs into an appropriate format
            groups_cons = ((group[0], str([group[0]]), group[1].name,
                           group[1].weight, group[1])
                           if group[1].link is None
                           else ("[{}]".format(", ".join([group[0]]+group[1].link)),
                                 str([group[0]]+group[1].link), group[1].name,
                                 group[1].weight, group[1])
                           for group in groups_cons)

            # Multiplicate the pairs generator
            groups_cons = tee(groups_cons, 3)

            return (tuple(r" $\rightarrow$ ".join((group[0], group[2]))
                          for group in groups_obj[0])
                    + tuple(r" $\rightarrow$ ".join((group[0], group[2]))
                            for group in groups_cons[0]),
                    tuple(divide(tracker[group[1]+'-'+group[2]], group[3])
                          for group in groups_obj[1])
                    + tuple(array(tracker[group[1]+'-'+group[2]])
                            for group in groups_cons[1]),
                    tuple(group[4] for group in groups_obj[2])
                    + tuple(group[4] for group in groups_cons[2]))

        # Determine the step length on the x-axis
        x_step = min(
            (5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000),
            key=lambda x: abs(ceil(
                max(len(track) for track in tracker.values())/x)-20))

        # Get the legend labels, the tracks, and the model objectives
        labels, tracks, model_objectives = get_labels_tracks_objectives()

        # Create a figure and a subplot
        figure, axis = subplots(figsize=(14, 8))

        # Loop over the number of tracks
        for i, _ in enumerate(tracks):

            # Check if the track belongs to the LKB model
            if 'Lyman-Kutcher-Burman NTCP' in labels[i]:

                # Plot the track
                axis.plot(range(1, tracks[i].size+1), tracks[i], '.-')

            # Check if the track belongs to the LQ Poisson TCP model
            if 'LQ Poisson TCP' in labels[i]:

                # Plot the track
                axis.plot(range(1, tracks[i].size+1), -tracks[i], '.-')

            # Check if the track belongs to the DT, EGB, KNN, NB or RF model
            if any(model_name in labels[i] for model_name in (
                    'Decision Tree', 'K-Nearest Neighbors', 'Naive Bayes',
                    'Random Forest')):

                # Check if the model predicts NTCP
                if 'NTCP' in model_objectives[i].name:

                    # Plot the track
                    axis.plot(range(1, tracks[i].size+1), tracks[i], '.-')

                else:

                    # Plot the track
                    axis.plot(range(1, tracks[i].size+1), -tracks[i], '.-')

            # Check if the track belongs to the LR or NN model
            if any(model_name in labels[i] for model_name in (
                    'Logistic Regression', 'Neural Network')):

                # Check if the model predicts NTCP
                if 'NTCP' in model_objectives[i].name:

                    # Plot the sigmoid-transformed track
                    axis.plot(range(1, tracks[i].size+1), sigmoid(
                        tracks[i], 1, 0), '.-')

                else:

                    # Plot the sigmoid-transformed inverse track
                    axis.plot(range(1, tracks[i].size+1), 1-sigmoid(
                        tracks[i], 1, 0), '.-')

            # Check if the track belongs to the SVM model
            elif 'Support Vector Machine' in labels[i]:

                # Get the SVM model
                svm = model_objectives[i].model.prediction_model

                if 'NTCP' in model_objectives[i].name:

                    # Plot the Platt-transformed track
                    axis.plot(range(1, tracks[i].size+1), sigmoid(
                        tracks[i], -svm.probA_, svm.probB_), '.-')

                else:

                    # Plot the Platt-transformed inverse track
                    axis.plot(range(1, tracks[i].size+1), 1-sigmoid(
                        tracks[i], -svm.probA_, svm.probB_), '.-')

        # Set x- and y-label
        axis.set_xlabel("Evaluation step", fontsize=11)
        axis.set_ylabel("(N)TCP", fontsize=11)

        # Change the tick label sizes for both axes
        axis.tick_params(axis='both', which='major', labelsize=9)

        # Set x- and y-ticks
        axis.set_xticks(tuple(i*x_step for i in range(
            int(ceil(max(len(track) for track in tracker.values())/x_step))+1)
            ))
        axis.set_yticks(tuple(i/20 for i in range(21)))

        # Set the font sizes for the tick labels
        for label in axis.get_xticklabels():
            label.set_fontsize(9)
        for label in axis.get_yticklabels():
            label.set_fontsize(9)

        # Set the x- and y-limits
        axis.set_xlim(0, max(len(track) for track in tracks)+1)
        axis.set_ylim(-0.01, 1.01)

        # Set the facecolor for the axis
        axis.set_facecolor("whitesmoke")

        # Specify the grid with a subgrid
        axis.grid(which='major', color='lightgray', linewidth=0.8)
        axis.grid(which='minor', color='lightgray', linestyle=':',
                  linewidth=0.5)
        axis.minorticks_on()

        # Set the legend and its facecolor
        legend = axis.legend(labels, fontsize=9, framealpha=1)
        legend.get_frame().set_facecolor('snow')

        # Apply a tight layout to the figure
        figure.tight_layout()

        # Get the figure manager
        figure_manager = get_current_fig_manager()

        # Set the window title
        figure_manager.set_window_title("pyanno4rt - iterative (N)TCP value "
                                        "plot")

        # Show the plot in screen size
        figure_manager.window.showMaximized()
