"""Feature graph widget."""

# Author: Tim Ortkamp

# %% External package import

from itertools import islice, cycle
from numpy import multiply
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QVBoxLayout, QWidget
from pyqtgraph import (
    colormap, mkPen, PlotDataItem, PlotWidget, setConfigOptions, TextItem,
    ViewBox)
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

        # 
        self.outcome_viewbox = ViewBox()
        self.plot_widget.scene().addItem(self.outcome_viewbox)
        self.plot_widget.getAxis('right').linkToView(self.outcome_viewbox)
        self.outcome_viewbox.setXLink(self.plot_widget.plotItem)

        # Initialize the feature histories, outcomes and display styles
        self.histories = None
        self.outcomes = None
        self.styles = None

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

        # Set the marker styles
        markers = tuple(islice(
            cycle(['s', 't', 'd', 'star', 'x']), len(outcomes)))

        # Set the colormap
        colors = colormap.get('Blues', 'matplotlib').getLookupTable(
            nPts=len(outcomes))

        # Set the line styles
        linestyles = tuple(islice(
            cycle([Qt.DashLine, Qt.DotLine, Qt.DashDotLine]), len(outcomes)))

        # Create a dictionary for the history styles
        self.styles = dict(
            zip(list(outcomes), tuple(zip(markers, colors, linestyles))))

        # Set the plot labels
        self.plot_widget.setLabels(
            left="Feature value", bottom="Evaluation step",
            right="Outcome prediction [%]", top=" ")

        # 
        self.plot_widget.getAxis('left').setPen('#ff7f0e')
        self.plot_widget.getAxis('left').setTextPen('#ff7f0e')

        # 
        self.plot_widget.getAxis('right').setPen('#f7fbff')
        self.plot_widget.getAxis('right').setTextPen('#f7fbff')

        # 
        self.plot_widget.getViewBox().sigResized.connect(self.update_views)
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

        # 
        self.reset_graph()

        # 
        model_name = self.parent.model_cbox.currentText()
        feature_name = self.parent.feature_cbox.currentText()

        # 
        history = self.histories[model_name][feature_name]
        outcome = self.outcomes[model_name]

        # Plot the feature history
        plot = self.plot_widget.plot(
            range(1, len(history)+1),
            history,
            pen=mkPen(color=QColor('#ff7f0e'), style=Qt.SolidLine, width=2),
            symbol='o',
            symbolSize=7,
            symbolBrush=[QColor('#ff7f0e') for i in range(len(history))],
            name=feature_name,
            clickable=True)
        
        # Set the QPen
        pen = mkPen(
            color=self.styles[model_name][1], style=self.styles[model_name][2],
            width=2)

        # 
        if self.parent.show_ntcp_check.isChecked():

            # 
            self.outcome_viewbox.addItem(PlotDataItem(
                list(range(1, len(outcome)+1)), multiply(outcome, 100),
                pen=pen, symbol=self.styles[model_name][0], symbolSize=7,
                symbolBrush=[
                    self.styles[model_name][1] for i in range(len(outcome))],
                name=model_name))

        # Set the plot limits
        self.plot_widget.plotItem.vb.setLimits(
            xMin=0, xMax=len(history)+1)

        # Enable the auto-range
        self.plot_widget.plotItem.vb.enableAutoRange()

        # Show the grid
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)

        # Show the axes
        self.plot_widget.getPlotItem().showAxis('bottom')
        self.plot_widget.getPlotItem().showAxis('top')
        self.plot_widget.getPlotItem().showAxis('left')
        self.plot_widget.getPlotItem().showAxis('right')

        # Add hoverable tooltip to the scatter points
        plot.scatter.opts.update(
            hoverable=True,
            tip='step: {x:0.0f}\nvalue: {y:0.4f}'.format)

        # 
        self.text = TextItem(
            f'Correlation: {round(pearsonr(history, outcome)[0], 4)}')
        self.plot_widget.addItem(self.text)
        self.text.setPos(0,0.5)

    def reset_graph(self):
        """Reset the component graph."""

        # Clear the plot graph
        self.plot_widget.clear()

        # 
        self.outcome_viewbox.clear()

        # Hide the axes
        self.plot_widget.getPlotItem().hideAxis('bottom')
        self.plot_widget.getPlotItem().hideAxis('top')
        self.plot_widget.getPlotItem().hideAxis('left')
        self.plot_widget.getPlotItem().hideAxis('right')
