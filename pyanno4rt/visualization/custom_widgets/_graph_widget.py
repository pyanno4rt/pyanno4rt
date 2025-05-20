"""Graph widget."""

# Author: Tim Ortkamp

# %% External package import

from itertools import islice, cycle
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QVBoxLayout, QWidget
from pyqtgraph import (colormap, InfiniteLine, mkPen, PlotWidget, SignalProxy)

# %% Class definition


class GraphWidget(QWidget):
    """."""

    def __init__(self, parent=None):

        # 
        self.parent = parent

        # Call the superclass constructor
        super().__init__()

        # Set the vertical layout for the DVH widget
        graph_layout = QVBoxLayout(self)

        # 
        self.plot_graph = PlotWidget()

        # 
        graph_layout.addWidget(self.plot_graph)

        self.plot_graph.getPlotItem().hideAxis('bottom')
        self.plot_graph.getPlotItem().hideAxis('top')
        self.plot_graph.getPlotItem().hideAxis('left')
        self.plot_graph.getPlotItem().hideAxis('right')
        self.plot_graph.getPlotItem().hideButtons()

        self.line_styles = None

        # Create vertical and horizontal infinite lines
        self.vertical_line = InfiniteLine(angle=90)
        self.horizontal_line = InfiniteLine(angle=0, movable=False)

        # Disable the pens for the lines
        self.vertical_line.setPen(None)
        self.horizontal_line.setPen(None)

        # Add the lines to the graph
        self.plot_graph.addItem(self.vertical_line, ignoreBounds=True)
        self.plot_graph.addItem(self.horizontal_line, ignoreBounds=True)

    def add_style_and_data(self, tracker):
        """."""

        # 
        self.tracker = tracker

        # 
        self.tracks = tuple(tracker)

        # Get the colormap
        colors = colormap.get('jet', 'matplotlib').getLookupTable(
            nPts=len(self.tracks))

        # Set the line styles
        line_styles = tuple(islice(
            cycle([Qt.SolidLine, Qt.DashLine, Qt.DotLine, Qt.DashDotLine]),
            len(self.tracks)))

        # Create a dictionary with the segment styles
        self.line_styles = dict(zip(
            self.tracks, tuple(zip(colors, line_styles))))

        self.plot_graph.getPlotItem().showAxis('bottom')
        self.plot_graph.getPlotItem().showAxis('top')
        self.plot_graph.getPlotItem().showAxis('left')
        self.plot_graph.getPlotItem().showAxis('right')

        self.plot_graph.showGrid(x=True, y=True, alpha=0.2)
        self.plot_graph.setLabels(left=" ", right=" ", top=" ", bottom=" ")

        # ax_right = self.plot_graph.getAxis('right')
        # ax_right.setTicks([])
        # ax_top = self.plot_graph.getAxis('top')
        # ax_top.setTicks([])

        # Set the signal proxy to update the crosshair at mouse moves
        self.crosshair_update = SignalProxy(
            self.plot_graph.scene().sigMouseMoved, rateLimit=60,
            slot=self.update_crosshair)

        # Set the graph title
        self.plot_graph.setTitle(
            "<span style='color: #FFAE42; "
            "font-size: 11pt'>dose/fx: %0.2f</span>, "
            "<span style='color: #FFAE42; "
            "font-size: 11pt'>vRel: %0.1f</span>"
            % (0, 0.0))

        # 
        self.plot_graph.plotItem.vb.setLimits(
            xMin=0, xMax=max(len(track) for track in tracker.values()))

        # 
        self.plot_graph.plotItem.vb.enableAutoRange()

    def reset_graph(self):
        """."""

        self.plot_graph.clear()
        self.plot_graph.getPlotItem().hideAxis('bottom')
        self.plot_graph.getPlotItem().hideAxis('top')
        self.plot_graph.getPlotItem().hideAxis('left')
        self.plot_graph.getPlotItem().hideAxis('right')
        self.plot_graph.setTitle(None)
        if hasattr(self, 'crosshair_update'):
            delattr(self, 'crosshair_update')

    def update_crosshair(
            self,
            event):
        """Update the crosshair at mouse moves."""

        # Get the coordinates from the triggered event
        coordinates = event[0]

        # Check if the coordinates lie within the scene bounding rectangle
        if self.plot_graph.sceneBoundingRect().contains(coordinates):

            # Get the mouse point in the view's coordinate system
            mouse_point = self.plot_graph.plotItem.vb.mapSceneToView(
                coordinates)

            # 
            limits = self.plot_graph.plotItem.vb.getState()['limits']

            if ((0 <= mouse_point.x() <= limits['xLimits'][1])
                    and (limits['yLimits'][0]
                         <= mouse_point.y()
                         <= limits['yLimits'][1])):

                # Update the graph title
                self.plot_graph.setTitle(
                    "<span style='color: #FFAE42; "
                    "font-size: 11pt'>dose/fx: "
                    "%0.2f</span>, <span style='color: #FFAE42; "
                    "font-size: 11pt'>vRel: %0.1f</span>"
                    % (mouse_point.x(), mouse_point.y()))

            else:

                # Update the graph title
                self.plot_graph.setTitle(
                    "<span style='color: #FFAE42; "
                    "font-size: 11pt'>dose/fx: "
                    "%0.2f</span>, <span style='color: #FFAE42; "
                    "font-size: 11pt'>vRel: %0.1f</span>"
                    % (0, 0.0))

            # Update the positions of vertical and horizontal lines
            self.vertical_line.setPos(mouse_point.x())
            self.horizontal_line.setPos(mouse_point.y())

    def update_graph(self):
        """."""

        for track in self.tracks:

            pen = mkPen(
                color=self.line_styles[track][0],
                style=self.line_styles[track][1],
                width=1)

            self.plot_graph.plot(
                range(len(self.tracker[track])), self.tracker[track], pen=pen,
                name=track, clickable=True)
