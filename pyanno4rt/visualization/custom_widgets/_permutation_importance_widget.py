"""Boxplot widget."""

# Author: Tim Ortkamp

# %% External package import

from matplotlib.cbook import boxplot_stats
from PyQt5.QtWidgets import QApplication, QVBoxLayout, QWidget
from pyqtgraph import (
    colormap, GraphicsObject, mkBrush, mkPen, PlotWidget, QtCore, QtGui,
    setConfigOptions)

# %% Internal package import

from pyanno4rt.tools import flatten
from pyanno4rt.visualization._custom_styles import tooltip

# %% Plotting options

setConfigOptions(antialias=True)

# %% Class definition


class BoxplotItem(GraphicsObject):
    """
    Boxplot item class.

    This class provides a boxplot shape for the permutation importance widget.

    Parameters
    ----------
    position : int
        Position of the boxplot on the x-axis.

    statistic : tuple
        Feature name and statistics for the boxplot.

    color : ndarray
        RGB values for the boxplot rectangle.
    """

    def __init__(
            self,
            position,
            statistic,
            color):

        # Call the superclass constructor
        GraphicsObject.__init__(self)

        # Get the position and color
        self.position = position
        self.color = color

        # Get the category name
        self.name = statistic[0]

        # Get the boxplot values
        (self.mean, self.median, self.lower_quartile, self.upper_quartile,
         self.lower_iqr, self.upper_iqr, self.outliers) = statistic[1]

        # Accept hover events
        self.setAcceptHoverEvents(True)

        # Generate the picture
        self.generatePicture()

    def hoverEnterEvent(
            self,
            _):
        """Show a tooltip with the statistics."""

        # Set the tooltip
        self.setToolTip(
            QApplication.translate(
                "MainWindow",
                f"<html><head/><body><p><b>{self.name}</b>"
                "<tr><td align='right'>x&#772; = </td>"
                f"<td>{round(self.mean, 4)}</td></tr>"
                "<tr><td align='right'>x<sub>min</sub> = </td>"
                f"<td>{round(self.lower_iqr, 4)}</td></tr>"
                "<tr><td>x<sub>0.25</sub> = </td>"
                f"<td>{round(self.lower_quartile, 4)}</td></tr>"
                "<tr><td align='right'>x<sub>0.5</sub> = </td>"
                f"<td>{round(self.median, 4)}</td></tr>"
                "<tr><td align='right'>x<sub>0.75</sub> = </td>"
                f"<td>{round(self.upper_quartile, 4)}</td></tr>"
                "<tr><td align='right'>x<sub>max</sub> = </td>"
                f"<td>{round(self.upper_iqr, 4)}</td></tr>"
                "<tr><td align='right'>n<sub>out</sub> = </td>"
                f"<td>{len(self.outliers)}</td></tr>"
                "</table></p></body></html>"))

    def generatePicture(self):
        """Generate the picture."""

        # Initialize the picture
        self.picture = QtGui.QPicture()

        # Initialize the painter
        pen = QtGui.QPainter(self.picture)

        # Set the white default pen
        pen.setPen(mkPen('w'))

        # Set the brush color
        pen.setBrush(mkBrush(self.color))

        # Check if the quartiles are not zero
        if (self.lower_quartile, self.upper_quartile) != (0.0, 0.0):

            # Draw the connecting line between the quartiles
            pen.drawLine(
                QtCore.QPointF(self.position+1, self.lower_iqr),
                QtCore.QPointF(self.position+1, self.upper_iqr))

        # Draw the boxplot rectangle
        pen.drawRect(QtCore.QRectF(
            self.position+2/3, self.lower_quartile, 2/3,
            self.upper_quartile-self.lower_quartile))

        # Draw the median line
        pen.drawLine(
            QtCore.QPointF(self.position+2/3, self.median),
            QtCore.QPointF(self.position+4/3, self.median))

        # Draw the lower whisker
        pen.drawLine(
            QtCore.QPointF(self.position+5/6, self.lower_iqr),
            QtCore.QPointF(self.position+7/6, self.lower_iqr))

        # Draw the upper whisker
        pen.drawLine(
            QtCore.QPointF(self.position+5/6, self.upper_iqr),
            QtCore.QPointF(self.position+7/6, self.upper_iqr))

        # Loop over the outliers
        for point in self.outliers:

            # Set the brush color to None
            pen.setBrush(mkBrush(None))

            # Draw a circular point
            pen.drawEllipse(
                QtCore.QPointF(self.position+1, point), 0.008, 0.008)

        # Deactivate the painter
        pen.end()

    def paint(
            self,
            pen,
            *args):
        """
        Paint the item's contents.

        Parameters
        ----------
        pen : object of class :class:`~PyQt5.QtGui.QPainter`
            The object used to perform drawing operations.

        args : tuple
            Additional (non-keyworded) parameters.
        """

        # Draw the picture
        pen.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        """Return the bounding rectangle of the item."""

        return QtCore.QRectF(
            self.position+2/3, min(flatten((self.outliers, self.lower_iqr))),
            2/3,
            max(flatten((self.outliers, self.upper_iqr)))
            - min(flatten((self.outliers, self.lower_iqr))))


class PermutationImportanceWidget(QWidget):
    """
    Permutation importance graph widget for the visualizer.

    This class sets up a permutation importance graph widget for the visual \
    analysis tool, including boxplots with the feature importance values.

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

        # Initialize the statistics and colors
        self.statistics = None
        self.colors = None

    def add_style_and_data(
            self,
            importances):
        """
        Add the widget style and backend data.

        Parameters
        ----------
        importances : dict
            Dictionary with the permutation importance values.
        """

        # Calculate the importance statistics
        self.statistics = self.calculate_statistics(importances)

        # Set the colormaps
        self.colors = {
            key: colormap.get('tab20b', 'matplotlib').getLookupTable(
                nPts=self.statistics[key]['number_of_features'])
            for key in self.statistics}

        # Show the grid
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)

        # Connect the model name field with the boxplot update
        self.parent.model_name_cbox.currentTextChanged.connect(
            self.update_boxplots)

        # Connect the domain field with the boxplot update
        self.parent.domain_cbox.currentTextChanged.connect(
            self.update_boxplots)

        # Connect the number of features field with the boxplot update
        self.parent.num_features_sbox.valueChanged.connect(
            self.update_boxplots)

    def calculate_statistics(
            self,
            importances):
        """
        Calculate the boxplot statistics from the importance values.

        Parameters
        ----------
        importances : dict
            Dictionary with the permutation importance values.

        Returns
        -------
        dict
            Dictionary with the model-wise boxplot statistics.
        """

        # Initialize the statistics dictionary
        statistics = {
            key: {'Training': None, 'Out-of-folds': None}
            for key in importances}

        # Loop over the importance results
        for key, value in importances.items():

            # Loop over the domains
            for domain in ('Training', 'Out-of-folds'):

                # Calculate and sort the importance statistics
                statistics[key][domain] = sorted((
                    (feature,
                     tuple(stats[key] for key in (
                        'mean', 'med', 'q1', 'q3', 'whislo', 'whishi',
                        'fliers')))
                    for feature, stats in zip(
                            self.parent.plan.datahub.datasets[key][
                                'feature_names'],
                            boxplot_stats(value[domain]))),
                    reverse=True, key=lambda x: x[1][0])

            # Add the number of features
            statistics[key]['number_of_features'] = len(
                statistics[key]['Training'])

        return statistics

    def update_boxplots(self):
        """Update the importance boxplots."""

        # Reset the boxplots
        self.reset_boxplots()

        # Get the current model name and domain
        model_name = self.parent.model_name_cbox.currentText()
        domain = self.parent.domain_cbox.currentText()

        # Get the current statistics and colors
        statistics = self.statistics[model_name][domain]
        colors = self.colors[model_name]
        top_k = self.parent.num_features_sbox.value()

        # Update the range for the top-k features
        self.parent.num_features_sbox.setRange(1, len(statistics))

        # Set the plot limits
        self.plot_widget.plotItem.vb.setLimits(
            xMin=0.6, xMax=len(statistics[:top_k])+0.4,
            yMin=min(flatten(
                (statistic[1][4].tolist(), statistic[1][6].tolist())
                for statistic in statistics[:top_k]))-0.05,
            yMax=max(flatten(
                (statistic[1][5].tolist(), statistic[1][6].tolist())
                for statistic in statistics[:top_k]))+0.05)

        # Enable the auto-range
        self.plot_widget.plotItem.vb.enableAutoRange()

        # Set the bottom axis ticks
        ax = self.plot_widget.getAxis('bottom')
        ax.setTicks([[
            (position+1, text) for position, text in enumerate(
                statistic[0] for statistic in statistics)]])

        # # Set the top axis ticks
        ax = self.plot_widget.getAxis('top')
        ax.setTicks([[
            (position+1, text) for position, text in enumerate(
                statistic[0] for statistic in statistics)]])

        # Loop over the boxplot inputs
        for position, (statistic, color) in enumerate(
                zip(statistics[:top_k], colors[:top_k])):

            # Initialize the boxplot item
            boxplot_item = BoxplotItem(position, statistic, color)

            # Add the item
            self.plot_widget.addItem(boxplot_item)

        # Show the axes
        self.plot_widget.getPlotItem().showAxis('bottom')
        self.plot_widget.getPlotItem().showAxis('top')
        self.plot_widget.getPlotItem().showAxis('left')
        self.plot_widget.getPlotItem().showAxis('right')

    def reset_boxplots(self):
        """Reset the importance boxplots."""

        # Clear the plot graph
        self.plot_widget.clear()

        # Hide the axes
        self.plot_widget.getPlotItem().hideAxis('bottom')
        self.plot_widget.getPlotItem().hideAxis('top')
        self.plot_widget.getPlotItem().hideAxis('left')
        self.plot_widget.getPlotItem().hideAxis('right')
