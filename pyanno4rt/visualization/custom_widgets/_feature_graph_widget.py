"""Feature graph widget."""

# Author: Tim Ortkamp

# %% External package import

from numpy import multiply
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QVBoxLayout, QWidget
from pyqtgraph import (
    mkPen, PlotDataItem, PlotWidget, setConfigOptions, TextItem, ViewBox)
from scipy.stats import pearsonr

# %% Internal package import

from pyanno4rt.visualization._custom_styles import tooltip

# %% Plotting options

setConfigOptions(antialias=True)

# %% Class definition


class FeatureGraphWidget(QWidget):
    """
    Feature graph widget for the visualizer.

    This class sets up a feature graph widget for the visual analysis tool, \
    including a line plot with the iterative feature values.

    Parameters
    ----------
    parent : object of class \
        :class:`~pyanno4rt.visualization._visualizer.Visualizer`, default=None
        The object representing the parent window for embedding.
    """

    def __init__(
            self,
            parent=None):

        # Call the superclass constructor
        super().__init__()

        # Get the parent window
        self.parent = parent

        # Set the vertical layout for the graph
        graph_layout = QVBoxLayout(self)

        # Initialize the plot widget
        self.plot_widget = PlotWidget()

        # Add margins to the widget
        self.plot_widget.plotItem.setContentsMargins(20, 10, 10, 20)

        # Set the style sheet for the tooltips
        self.plot_widget.setStyleSheet(tooltip)

        # Add the widget to the layout
        graph_layout.addWidget(self.plot_widget)

        # Initialize the viewbox for the outcome values
        self.outcome_viewbox = ViewBox()

        # Add the viewbox to the widget
        self.plot_widget.scene().addItem(self.outcome_viewbox)

        # Add the text item to the widget
        self.text = TextItem(anchor=(0, 0), ensureInBounds=True)
        self.text.setFlag(self.text.GraphicsItemFlag.ItemIgnoresTransformations)
        self.text.setParentItem(self.plot_widget.plotItem)

        # Initialize the feature histories and outcomes
        self.histories, self.outcomes = None, None

    def add_style_and_data(
            self,
            histories,
            outcomes):
        """
        Add the feature history and iterative outcome data.

        Parameters
        ----------
        histories : dict
            Dictionary with the iteration-wise model feature values.

        outcomes : dict
            Dictionary with the teration-wise model outcome values.
        """

        # Set the feature histories and outcomes
        self.histories = histories
        self.outcomes = outcomes

        # Set the plot labels
        self.plot_widget.setLabels(
            bottom="Evaluation step", top=" ")

        # Connect the fields with the event signals
        self.parent.model_cbox.currentTextChanged.connect(self.update_graph)
        self.parent.feature_cbox.currentTextChanged.connect(self.update_graph)
        self.parent.show_ntcp_check.stateChanged.connect(self.update_graph)

    def update_views(self):
        """Updates the views of the plot to keep the right y-axis in sync."""

        # Match the geometry of the right viewbox with the main viewbox
        self.outcome_viewbox.setGeometry(
            self.plot_widget.getViewBox().sceneBoundingRect())

        # Update the linked axes
        self.outcome_viewbox.linkedViewChanged(
            self.plot_widget.getViewBox(), self.outcome_viewbox.XAxis)

    def update_graph(self):
        """Update the feature graph."""

        # Reset the graph
        self.reset_graph()

        # Get the model and feature name
        model_name = self.parent.model_cbox.currentText()
        feature_name = self.parent.feature_cbox.currentText()

        # Get the feature history and the outcome values for the model
        history = self.histories[model_name][feature_name]
        outcome = self.outcomes[model_name]

        # Check if the (N)TCP should be displayed
        if self.parent.show_ntcp_check.isChecked():

            # Align the right y-axis
            self.plot_widget.getAxis('right').linkToView(self.outcome_viewbox)

            # Align the x-axis
            self.outcome_viewbox.setXLink(self.plot_widget.plotItem)

            # Connect the viewbox with the update method
            self.plot_widget.getViewBox().sigResized.connect(self.update_views)

            # Set the colors for the split y-axis
            ocolor = '#1f77b4'
            hcolor = '#e7ba52'

            # Plot the outcome values
            self.outcome_viewbox.addItem(PlotDataItem(
                list(range(1, len(outcome)+1)),
                multiply(outcome, 100),
                pen=mkPen(color=QColor(ocolor), style=Qt.DashLine, width=2),
                symbol='s',
                symbolSize=7,
                symbolBrush=[ocolor for i in range(len(outcome))],
                name=model_name))

            # Show the feature-outcome correlation
            self.text.setText(
                f'ρ={round(pearsonr(history, outcome)[0], 4)}',
                color='#ad494a')

            # Set the label for the right y-axis
            self.plot_widget.setLabel('right', 'Outcome prediction [%]')

            # Loop over the y-axes and colors
            for side, color in (('left', hcolor), ('right', ocolor)):

                # Set the pen and text color
                self.plot_widget.getAxis(side).setPen(color)
                self.plot_widget.getAxis(side).setTextPen(color)

        else:

            # Set the purple color for the feature y-axis
            hcolor = '#393b79'

        # Plot the feature history
        plot = self.plot_widget.plot(
            range(1, len(history)+1),
            history,
            pen=mkPen(color=QColor(hcolor), style=Qt.SolidLine, width=2),
            symbol='o',
            symbolSize=7,
            symbolBrush=[QColor(hcolor) for i in range(len(history))],
            name=feature_name,
            clickable=True)

        # Override the left y-axis label with the feature name
        self.plot_widget.setLabel('left', feature_name)

        # Set the plot limits
        self.plot_widget.plotItem.vb.setLimits(xMin=0, xMax=len(history)+1)

        # Enable the auto-range
        self.plot_widget.plotItem.vb.enableAutoRange()

        # Show the grid
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)

        # Loop over the axes
        for axis in ('bottom', 'top', 'left', 'right'):

            # Show the axis
            self.plot_widget.getPlotItem().showAxis(axis)

        # Add hoverable tooltip to the scatter points of the feature curve
        plot.scatter.opts.update(
            hoverable=True, tip='step: {x:0.0f}\nvalue: {y:0.4f}'.format)

    def reset_graph(self):
        """Reset the feature graph."""

        # Clear the plot graph
        self.plot_widget.clear()

        # Clear the outcome viewbox
        self.outcome_viewbox.clear()

        # Re-align the right y-axis to the default
        self.plot_widget.getAxis('right').linkToView(
            self.plot_widget.getViewBox())

        # Reset the title
        self.plot_widget.setTitle('')

        # Reset the right y-axis label
        self.plot_widget.setLabel('right', '')

        # Loop over the y-axes
        for side in ('left', 'right'):

            # Reset the pen and text color
            self.plot_widget.getAxis(side).setPen()
            self.plot_widget.getAxis(side).setTextPen()

        # Loop over the axes
        for axis in ('bottom', 'top', 'left', 'right'):

            # Show the axis
            self.plot_widget.getPlotItem().hideAxis(axis)

        # Reset the text item
        self.text.setText('')
