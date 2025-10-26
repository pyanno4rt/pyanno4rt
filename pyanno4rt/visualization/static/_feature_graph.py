"""Iterative feature graph."""

# Author: Tim Ortkamp

# %% External package import

from matplotlib.pyplot import get_current_fig_manager, subplots
from numpy import ceil, floor, multiply
from scipy.stats import pearsonr

# %% Internal package import

from pyanno4rt.tools import filter_dict

# %% Class definition


class FeatureGraph():
    """
    Iterative feature graph class.

    This class provides a plot with the iteration-wise values of the \
    model features.

    Parameters
    ----------
    title : str, default=''
        Title of the plot.

    titlesize : int, default=16
        Font size of the title.

    xlabel : str, default='Evaluation step'
        Label for the x-axis.

    ylabel : str, default='Feature value',
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
            ylabel='Feature value',
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
            history,
            outcome):
        """
        Open the iterative feature graph.

        Parameters
        ----------
        history : tuple
            Tuple with the feature name and iterative feature values.

        outcome : tuple
            Tuple with the model name and iterative outcome values.
        """

        # Get the input variables
        feature_name, feature_values = history
        model_name, outcome_values = outcome

        # Get the feature statistics
        feature_max = max(feature_values)
        feature_min = min(feature_values)
        feature_len = len(feature_values)

        # Set the expected number of ticks
        number_of_ticks = 20 if self.ticksize < 17 else 10

        # Get the figure and axis objects
        figure, axis = subplots(figsize=(14, 8))

        # Set the plot title
        axis.set_title(
            label=self.title, fontsize=self.titlesize, fontweight='semibold',
            pad=10)

        # Check if outcome values are provided
        if outcome_values is not None:

            # Get the outcome statistics
            outcome_max = max(outcome_values)
            outcome_min = min(outcome_values)

            # Set the colors for the split y-axis
            ocolor = '#1f77b4'
            hcolor = '#e7ba52'

            # Determine the step length on the second y-axis
            y_step_out = min(
                sorted(base*10**i for base in (1, 2, 5) for i in range(-6, 6)),
                key=lambda x: abs(
                    ceil((outcome_max-outcome_min)/x)-number_of_ticks))

            # Create the second y-axis
            axis2 = axis.twinx()

            # Plot the outcome values
            axis2.plot(
                range(1, len(outcome_values)+1),
                multiply(outcome_values, 100),
                marker='s',
                markersize=1.5*self.linewidth,
                color=ocolor,
                linestyle='--',
                linewidth=0.5*self.linewidth)

            # Set the y-labels
            axis.set_ylabel(
                ylabel=(
                    self.ylabel if self.ylabel != 'Feature value'
                    else feature_name),
                fontsize=self.labelsize, color=hcolor)
            axis2.set_ylabel('Outcome prediction [%]', color=ocolor)

            # Set the axis colors
            axis.tick_params(axis='y', colors=hcolor)
            axis2.tick_params(axis='y', colors=ocolor)

            # Set the y-ticks
            axis.set_yticks(tuple(
                i*y_step_out for i in range(
                    int(floor(outcome_min/y_step_out))-1,
                    int(ceil(outcome_max/y_step_out))+1)))

            # Set the y-limits
            axis.set_ylim(outcome_min-y_step_out/2, outcome_max+y_step_out/2)

            # Add the correlation value
            figure.text(
                .01, .99,
                f'ρ={round(pearsonr(feature_values, outcome_values)[0], 4)}',
                ha='left', va='top', transform=axis.transAxes, color='#ad494a')

        else:

            # Set the purple color for the first y-axis
            hcolor = '#393b79'

            # Set the first y-label
            axis.set_ylabel(
                ylabel=(
                    self.ylabel if self.ylabel != 'Feature value'
                    else feature_name),
                fontsize=self.labelsize)

        # Determine the step length on the shared x-axis
        x_step = min(
            sorted(base*10**i for base in (1, 2, 5) for i in range(6)),
            key=lambda x: abs(ceil(feature_len/x)-number_of_ticks))

        # Determine the step length on the first y-axis
        y_step = min(
            sorted(base*10**i for base in (1, 2, 5) for i in range(-6, 6)),
            key=lambda x: abs(
                ceil((feature_max-feature_min)/x)-number_of_ticks))

        # Plot the feature values
        axis.plot(
            range(1, len(feature_values)+1),
            feature_values,
            marker='o',
            markersize=1.5*self.linewidth,
            color=hcolor,
            linestyle='-',
            linewidth=0.5*self.linewidth)

        # Set the x-label
        axis.set_xlabel(xlabel=self.xlabel, fontsize=self.labelsize)

        # Configure the axis ticks
        axis.tick_params(axis='both', which='major', labelsize=self.ticksize)

        # Set the x- and y-ticks
        axis.set_xticks(tuple(
            i*x_step for i in range(int(ceil(feature_len/x_step))+1)))
        axis.set_yticks(tuple(
            i*y_step for i in range(
                int(floor(feature_min/y_step))-1,
                int(ceil(feature_max/y_step))+1)))

        # Set the x- and y-limits
        axis.set_xlim(0, feature_len+x_step/2)
        axis.set_ylim(feature_min-y_step/2, feature_max+y_step/2)

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

        # Apply a tight layout
        figure.tight_layout()

        # Get the figure manager
        figure_manager = get_current_fig_manager()

        # Set the window title
        figure_manager.set_window_title("Iterative feature graph")

        # Show the full-screen plot
        figure_manager.window.showMaximized()
