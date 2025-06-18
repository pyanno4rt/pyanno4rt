"""Component graph widget."""

# Author: Tim Ortkamp

# %% External package import

from itertools import islice, cycle
from numpy import ceil
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QVBoxLayout, QWidget
from pyqtgraph import colormap, InfiniteLine, mkPen, PlotWidget, SignalProxy

# %% Internal package import

from pyanno4rt.tools import flatten

# %% Class definition


class ComponentGraphWidget(QWidget):
    """
    Component graph widget for the visualizer.

    This class sets up a component graph widget for the visual analysis tool, \
    including a line plot with the iterative component values.

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

        # Generate the vertical and horizontal infinite lines
        self.vertical_line = InfiniteLine(angle=90)
        self.horizontal_line = InfiniteLine(angle=0)

        # Add the lines to the widget
        self.plot_widget.addItem(self.vertical_line)
        self.plot_widget.addItem(self.horizontal_line)

        # Add the widget to the layout
        graph_layout.addWidget(self.plot_widget)

        # Initialize the component tracker and the display styles
        self.tracker = None
        self.styles = None

        # Initialize the crosshair update
        self.crosshair_update = None

    def add_style_and_data(
            self,
            tracker):
        """
        Add the display styles and tracker data.

        Parameters
        ----------
        tracker : dict
            Dictionary with the iteration-wise plan component values.
        """

        # Set the tracker
        self.tracker = tracker

        # Get the track statistics
        track_min = min(flatten(tracker.values()))
        track_max = max(flatten(tracker.values()))
        track_num = len(tracker)

        # Set the marker styles
        markers = tuple(islice(
            cycle(['o', 's', 't', 'd', 'star', 'x']), track_num))

        # Set the colormap
        colors = colormap.get('tab20b', 'matplotlib').getLookupTable(
            nPts=track_num)

        # Set the line styles
        linestyles = tuple(islice(
            cycle([Qt.SolidLine, Qt.DashLine, Qt.DotLine, Qt.DashDotLine]),
            track_num))

        # Create a dictionary for the track styles
        self.styles = dict(
            zip(self.tracker, tuple(zip(markers, colors, linestyles))))

        # Set the plot title
        self.plot_widget.setTitle(
            "<span style='color: #FFAE42; font-size: 10pt'>"
            "step: %0.0f, value: %0.2f</span>" % (0, 0.00))

        # Set the plot labels
        self.plot_widget.setLabels(
            left="Component value", bottom="Evaluation step",
            right=" ", top=" ")

        # Determine the step length on the y-axis
        y_step = min(
            sorted(base*10**i for base in (1, 2, 5) for i in range(-6, 6)),
            key=lambda x: abs(ceil((track_max-track_min)/x)-20))

        # Set the plot limits
        self.plot_widget.plotItem.vb.setLimits(
            xMin=0, xMax=max(len(track) for track in tracker.values()),
            yMin=track_min-y_step, yMax=track_max+y_step)

        # Enable the auto-range
        self.plot_widget.plotItem.vb.enableAutoRange()

        # Add the legend
        self.plot_widget.addLegend(
            offset=(-0.2, 0.2),
            labelTextSize=(
                '9pt' if track_num <= 13 else '6pt' if track_num <= 23
                else '3pt' if track_num <= 33 else '0pt'))

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
        Select a component curve.

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

                    # Increase the symbol size
                    event.setSymbolSize(14)

                    # Increase the pen width
                    event.curve.setPen(mkPen(
                        color=pen.color(), style=pen.style(), width=5))

                    # Get the optimization data
                    optimization = self.parent.plan.datahub.optimization

                    # Get the optimization components
                    components = (
                        optimization['problem'].objectives
                        | optimization['problem'].constraints)

                    # Get the selected instance attribute
                    component_type, embedding, weight, rank, bounds = (getattr(
                        components[event.name()]['instance'], attribute)
                        for attribute in (
                            'component_type', 'embedding', 'weight', 'rank',
                            'bounds'))

                    # Display the component name
                    self.parent.comp_component_ledit.setText(event.name())

                    # Display the component type
                    self.parent.comp_type_ledit.setText(component_type)

                    # Display the component embedding mode
                    self.parent.comp_embedding_ledit.setText(embedding)

                    # Display the component weight
                    self.parent.comp_weight_ledit.setText(str(weight))

                    # Display the component rank
                    self.parent.comp_rank_ledit.setText(str(rank))

                    # Display the component bounds
                    self.parent.comp_bounds_ledit.setText(str(bounds))

                else:

                    # Decrease the symbol size
                    event.setSymbolSize(7)

                    # Decrease the pen width
                    event.curve.setPen(mkPen(
                        color=pen.color(), style=pen.style(), width=2))

                    # Clear the component line editors
                    self.parent.comp_component_ledit.clear()
                    self.parent.comp_type_ledit.clear()
                    self.parent.comp_embedding_ledit.clear()
                    self.parent.comp_weight_ledit.clear()
                    self.parent.comp_rank_ledit.clear()
                    self.parent.comp_bounds_ledit.clear()

            else:

                # Keep the symbol size
                item.setSymbolSize(7)

                # Keep the pen width
                item.curve.setPen(
                    mkPen(color=pen.color(), style=pen.style(), width=2))

    def unselect_curve(
            self,
            event):
        """
        Unselect a component curve.

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

                # Reset the symbol size
                item.setSymbolSize(7)

            # Clear the component line editors
            self.parent.comp_component_ledit.clear()
            self.parent.comp_type_ledit.clear()
            self.parent.comp_embedding_ledit.clear()
            self.parent.comp_weight_ledit.clear()
            self.parent.comp_rank_ledit.clear()
            self.parent.comp_bounds_ledit.clear()

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
                    and (limits['yLimits'][0]
                         <= mouse_point.y()
                         <= limits['yLimits'][1])):

                # Set the point to the current mouse point
                point = (mouse_point.x(), mouse_point.y())

            else:

                # Set the point to default
                point = (0, 0.00)

            # Update the graph title
            self.plot_widget.setTitle(
                "<span style='color: #FFAE42; font-size: 10pt'>"
                "step: %0.0f, value: %0.2f</span>" % point)

    def update_graph(self):
        """Update the component graph."""

        # Get the maximum track length
        track_len = max(len(track) for track in self.tracker.values())

        # Loop over the tracks
        for track in self.tracker:

            # Set the QPen
            pen = mkPen(
                color=self.styles[track][1], style=self.styles[track][2],
                width=2)

            # Plot the track
            plot = self.plot_widget.plot(
                range(len(self.tracker[track])),
                self.tracker[track],
                pen=pen,
                symbol=self.styles[track][0],
                symbolSize=7,
                symbolBrush=[self.styles[track][1] for i in range(track_len)],
                name=track,
                clickable=True)

            # Connect the plot with the curve selection methods
            plot.sigClicked.connect(self.select_curve)
            self.plot_widget.scene().sigMouseClicked.connect(
                self.unselect_curve)

    def reset_graph(self):
        """Reset the component graph."""

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

        # Clear the component line editors
        self.parent.comp_component_ledit.clear()
        self.parent.comp_type_ledit.clear()
        self.parent.comp_embedding_ledit.clear()
        self.parent.comp_weight_ledit.clear()
        self.parent.comp_rank_ledit.clear()
        self.parent.comp_bounds_ledit.clear()
