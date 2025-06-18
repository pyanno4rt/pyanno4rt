"""DVH graph widget."""

# Author: Tim Ortkamp

# %% External package import

from itertools import islice, cycle
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QVBoxLayout, QWidget
from pyqtgraph import colormap, InfiniteLine, mkPen, PlotWidget, SignalProxy

# %% Class definition


class DVHGraphWidget(QWidget):
    """
    DVH graph widget for the graphical user interface.

    This class sets up a DVH graph widget for the graphical user interface, \
    including a line plot with the segment-wise DVH values.

    Parameters
    ----------
    parent : object of class \
        :class:`~pyanno4rt.gui.windows._main_window.MainWindow`, default=None
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

        # Generate the vertical and horizontal infinite lines
        self.vertical_line = InfiniteLine(angle=90)
        self.horizontal_line = InfiniteLine(angle=0)

        # Add the lines to the widget
        self.plot_widget.addItem(self.vertical_line)
        self.plot_widget.addItem(self.horizontal_line)

        # Add the widget to the layout
        graph_layout.addWidget(self.plot_widget)

        # Initialize the segment names, the DVH data and the display styles
        self.segments = None
        self.dose_histogram = None
        self.styles = None

        # Initialize the crosshair update
        self.crosshair_update = None

    def add_style_and_data(
            self,
            dose_histogram):
        """
        Add the display styles and DVH data.

        Parameters
        ----------
        dose_histogram : dict
            Dictionary with information on the cumulative or differential \
            dose-volume histogram for each segment.
        """

        # Set the DVH data
        self.dose_histogram = dose_histogram

        # Get the segment names
        self.segments = tuple(
            segment for segment in (*dose_histogram,)
            if segment in dose_histogram['display_segments'])

        # Set the colormap
        colors = colormap.get('tab20b', 'matplotlib').getLookupTable(
            nPts=len(self.segments))

        # Set the line styles
        linestyles = tuple(islice(
            cycle([Qt.SolidLine, Qt.DashLine, Qt.DotLine, Qt.DashDotLine]),
            len(self.segments)))

        # Create a dictionary for the DVH styles
        self.styles = dict(
            zip(self.segments, tuple(zip(colors, linestyles))))

        # Set the plot title
        self.plot_widget.setTitle(
            "<span style='color: #FFAE42; font-size: 10pt'>"
            "dose: %0.2f, volume: %0.2f</span>" % (0.00, 0.00))

        # Set the plot labels
        self.plot_widget.setLabels(
            left="Relative volume [%]", bottom="Dose [Gy]",
            right=" ", top=" ")

        # Set the plot limits
        self.plot_widget.plotItem.vb.setLimits(
            xMin=0, xMax=dose_histogram['evaluation_points'][-1],
            yMin=-1, yMax=101)

        # Enable the auto-range
        self.plot_widget.plotItem.vb.enableAutoRange()

        # Add the legend
        self.plot_widget.addLegend(
            offset=(-0.2, 0.2),
            labelTextSize=(
                '9pt' if len(self.segments) <= 13
                else '6pt' if len(self.segments) <= 23
                else '3pt' if len(self.segments) <= 33
                else '0pt'))

        # Show the grid
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)

        # Set the signal proxy to update the crosshair at mouse moves
        self.crosshair_update = SignalProxy(
            self.plot_widget.scene().sigMouseMoved, rateLimit=60,
            slot=self.update_crosshair)

    def select_curve(
            self,
            event):
        """
        Select a DVH curve.

        Parameters
        ----------
        event : object of class :class:`~PyQt5.QtCore.QEvent`
            The object representing the event.
        """

        # Get all plot items
        items = self.plot_widget.getPlotItem().listDataItems()

        # Loop over the items
        for item in items:

            # Get the pen
            pen = item.curve.opts['pen']

            # Check if the current item triggers the event
            if item == event:

                # Construct the QPen
                pen = mkPen(pen)

                # Check if the pen width is 2
                if pen.width() == 2:

                    # Increase the pen width
                    event.curve.setPen(mkPen(
                        color=pen.color(), style=pen.style(), width=5))

                    # Get the dosimetrics data
                    dosimetrics = (
                        self.parent.plans[self.parent.plan_ledit.text()]
                        .datahub.dosimetrics)

                    # Display the segment name
                    self.parent.segment_ledit.setText(event.name())

                    # Display the segment mean dose
                    self.parent.mean_ledit.setText(str(
                        round(dosimetrics[event.name()]['mean'], 2)))

                    # Display the segment dose deviation
                    self.parent.std_ledit.setText(str(
                        round(dosimetrics[event.name()]['std'], 2)))

                    # Display the segment maximum dose
                    self.parent.maximum_ledit.setText(str(
                        round(dosimetrics[event.name()]['max'], 2)))

                    # Display the segment minimum dose
                    self.parent.minimum_ledit.setText(str(
                        round(dosimetrics[event.name()]['min'], 2)))

                else:

                    # Decrease the pen width
                    event.curve.setPen(mkPen(
                        color=pen.color(), style=pen.style(), width=2))

                    # Clear the dosimetrics line editors
                    self.parent.segment_ledit.clear()
                    self.parent.mean_ledit.clear()
                    self.parent.std_ledit.clear()
                    self.parent.maximum_ledit.clear()
                    self.parent.minimum_ledit.clear()

            else:

                # Keep the pen width
                item.curve.setPen(
                    mkPen(color=pen.color(), style=pen.style(), width=2))

    def unselect_curve(
            self,
            event):
        """
        Unselect a DVH curve.

        Parameters
        ----------
        event : object of class :class:`~PyQt5.QtCore.QEvent`
            The object representing the event.
        """

        # Check if the selection should be cleared
        if not event.isAccepted():

            # Get all plot items
            items = self.plot_widget.getPlotItem().listDataItems()

            # Loop over the items
            for item in items:

                # Get the pen
                pen = item.curve.opts['pen']

                # Reset the pen width
                item.curve.setPen(
                    mkPen(color=pen.color(), style=pen.style(), width=2))

            # Clear the dosimetrics line editors
            self.parent.segment_ledit.clear()
            self.parent.mean_ledit.clear()
            self.parent.std_ledit.clear()
            self.parent.maximum_ledit.clear()
            self.parent.minimum_ledit.clear()

    def update_crosshair(
            self,
            event):
        """
        Update the crosshair at mouse moves.

        Parameters
        ----------
        event : object of class :class:`~PyQt5.QtCore.QEvent`
            The object representing the event.
        """

        # Get the event coordinates
        coordinates = event[0]

        # Check if the coordinates are within the scene bounding rectangle
        if self.plot_widget.sceneBoundingRect().contains(coordinates):

            # Get the mouse point in the viewbox coordinate system
            mouse_point = self.plot_widget.plotItem.vb.mapSceneToView(
                coordinates)

            # Get the limits of the viewbox
            limits = self.plot_widget.plotItem.vb.getState()['limits']

            # Update the positions of vertical and horizontal lines
            self.vertical_line.setPos(mouse_point.x())
            self.horizontal_line.setPos(mouse_point.y())

            # Check if the mouse point is within the limits
            if ((0 <= mouse_point.x() <= limits['xLimits'][1])
                    and 0 <= mouse_point.y() <= 100):

                # Set the point to the current mouse point
                point = (mouse_point.x(), mouse_point.y())

            else:

                # Set the point to default
                point = (0.00, 0.00)

            # Update the graph title
            self.plot_widget.setTitle(
                "<span style='color: #FFAE42; font-size: 10pt'>"
                "dose: %0.2f, volume: %0.2f</span>" % point)

    def update_graph(self):
        """Update the DVH graph."""

        # Loop over the segment names
        for segment in self.segments:

            # Set the QPen
            pen = mkPen(
                color=self.styles[segment][0], style=self.styles[segment][1],
                width=2)

            # Plot the track
            plot = self.plot_widget.plot(
                self.dose_histogram['evaluation_points'],
                100*self.dose_histogram[segment]['dvh_values'],
                pen=pen,
                name=segment,
                clickable=True)

            # Connect the plot with the curve selection methods
            plot.sigClicked.connect(self.select_curve)
            self.plot_widget.scene().sigMouseClicked.connect(
                self.unselect_curve)

    def reset_graph(self):
        """Reset the DVH graph."""

        # Clear the plot graph
        self.plot_widget.clear()

        # Hide the axes
        self.plot_widget.getPlotItem().hideAxis('bottom')
        self.plot_widget.getPlotItem().hideAxis('top')
        self.plot_widget.getPlotItem().hideAxis('left')
        self.plot_widget.getPlotItem().hideAxis('right')

        # Clear the title
        self.plot_widget.setTitle(None)

        # Check if the widget has a crosshair update attribute
        if hasattr(self, 'crosshair_update'):

            # Remove the attribute
            delattr(self, 'crosshair_update')

        # Clear the dosimetrics line editors
        self.parent.segment_ledit.clear()
        self.parent.mean_ledit.clear()
        self.parent.std_ledit.clear()
        self.parent.maximum_ledit.clear()
        self.parent.minimum_ledit.clear()
