"""Permutation importance boxplots."""

# Author: Tim Ortkamp

# %% External package import

from matplotlib.pyplot import get_cmap, get_current_fig_manager, subplots
from numpy import linspace
from pandas import DataFrame
from seaborn import boxplot

# %% Internal package import

from pyanno4rt.tools import filter_dict

# %% Class definition


class PermutationImportanceBoxplot():
    """
    Permutation importance boxplots class.

    This class provides (sorted) boxplots with the permutation importance \
    statistics of the outcome prediction models.

    Parameters
    ----------
    title : str, default=''
        Title of the plot.

    titlesize : int, default=16
        Font size of the title.

    xlabel : str, default='Feature'
        Label for the x-axis.

    labelsize : int, default=11
        Font size of the labels.

    ticksize : int, default=9
        Font size of the axis ticks.

    tickangle : int, default=15
        Rotation angle of the axis ticks.

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

    labelsize : int
        See 'Parameters'.

    ticksize : int
        See 'Parameters'.

    tickangle : int
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
            xlabel='Feature',
            labelsize=11,
            ticksize=9,
            tickangle=15,
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
            model_name,
            domain,
            number_of_features):
        """
        Open the permutation importance boxplots.

        Parameters
        ----------
        treatment_plan : object of class \
            :class:`~pyanno4rt.base._treatment_plan.TreatmentPlan`
            The object used to represent the treatment plan.

        model_name : str
            Identifier for the outcome model.

        domain : {'Single-run', 'Multi-run'}
            Domain of the permutation importance statistics.

        number_of_features : int
            Number of features to be displayed.
        """

        # Get the model
        model = next((
            model for model in treatment_plan.data_model_handler.models
            if model.label == model_name), None)

        # Get the plot data
        data = list((
            model_name,
            model.inspector.results['permutation_importances']['score'],
            DataFrame(
                data=model.inspector.results['permutation_importances'][domain],
                columns=model.dataset.feature_names)))

        # Preprocess the permutation importance values
        data[2] = data[2].reindex(
            data[2].mean().sort_values(ascending=False).index,
            axis=1).iloc[:, :number_of_features]

        # Set the colormap
        colors = get_cmap('tab20b')(
            linspace(0, 1.0, len(model.dataset.feature_names)))

        # Get the figure and axis objects
        figure, axis = subplots(figsize=(14, 8))

        # Set the plot title
        axis.set_title(
            label=(f'{data[0]} ({domain}): top-{number_of_features} features'
                   if self.title == '' else self.title),
            fontsize=self.titlesize, fontweight='semibold', pad=10)

        # Plot the permutation importance data
        boxplot(
            data=data[2], palette=colors[:number_of_features].tolist(),
            ax=axis)

        # Set the x- and y-labels
        axis.set_xlabel(xlabel=self.xlabel, fontsize=self.labelsize)
        axis.set_ylabel(ylabel=f'Δ {data[1]}', fontsize=self.labelsize)

        # Configure the axis ticks
        axis.tick_params(axis='both', which='major', labelsize=self.ticksize)

        # Rotate the tick labels for the x-axis
        axis.tick_params(axis='x', labelrotation=self.tickangle)

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

        # Hide the axis behind the boxplots
        axis.set_axisbelow(True)

        # Apply a tight layout
        figure.tight_layout()

        # Get the figure manager
        figure_manager = get_current_fig_manager()

        # Set the window title
        figure_manager.set_window_title("Permutation importance boxplots")

        # Show the full-screen plot
        figure_manager.window.showMaximized()
