"""DVH widget."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from itertools import islice, cycle
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QVBoxLayout, QWidget
from pyqtgraph import (colormap, InfiniteLine, mkPen, PlotWidget, SignalProxy)

# %% Class definition


class DVHWidget(QWidget):
    """."""

    def __init__(self, parent=None):

        # 
        self.parent = parent

        # Call the superclass constructor
        super().__init__()

        # Set the vertical layout for the DVH widget
        dvh_layout = QVBoxLayout(self)
        dvh_layout.setContentsMargins(10, 10, 10, 0)

        # 
        self.plot_graph = PlotWidget()

        # 
        dvh_layout.addWidget(self.plot_graph)

        self.plot_graph.getPlotItem().hideAxis('bottom')
        self.plot_graph.getPlotItem().hideAxis('top')
        self.plot_graph.getPlotItem().hideAxis('left')
        self.plot_graph.getPlotItem().hideAxis('right')
        self.plot_graph.getPlotItem().hideButtons()

        self.segments = None
        self.dose_histogram = None
        self.segment_styles = None

        # Create vertical and horizontal infinite lines
        self.vertical_line = InfiniteLine(angle=90)
        self.horizontal_line = InfiniteLine(angle=0, movable=False)

        # Disable the pens for the lines
        self.vertical_line.setPen(None)
        self.horizontal_line.setPen(None)

        # Add the lines to the graph
        self.plot_graph.addItem(self.vertical_line, ignoreBounds=True)
        self.plot_graph.addItem(self.horizontal_line, ignoreBounds=True)

    def add_style_and_data(self, dose_histogram):
        """."""

        # 
        self.dose_histogram = dose_histogram

        # 
        self.segments = tuple(segment for segment in (*dose_histogram,)
                              if segment not in ('evaluation_points',
                                                 'display_segments'))

        # Get the colormap
        colors = colormap.get('jet', 'matplotlib').getLookupTable(
            nPts=len(self.segments))

        # Set the line styles
        line_styles = tuple(islice(cycle([Qt.SolidLine, Qt.DashLine,
                                          Qt.DotLine, Qt.DashDotLine]),
                                   len(self.segments)))

        # Create a dictionary with the segment styles
        self.segment_styles = dict(
            zip(self.segments, tuple(zip(colors, line_styles))))

        self.plot_graph.getPlotItem().showAxis('bottom')
        self.plot_graph.getPlotItem().showAxis('top')
        self.plot_graph.getPlotItem().showAxis('left')
        self.plot_graph.getPlotItem().showAxis('right')

        self.plot_graph.showGrid(x=True, y=True, alpha=0.2)
        self.plot_graph.setLabels(
            left=" ", right=" ", top=" ", bottom=" ")

        # ax_right = self.plot_graph.getAxis('right')
        # ax_right.setTicks([])
        # ax_top = self.plot_graph.getAxis('top')
        # ax_top.setTicks([])

        # Set the signal proxy to update the crosshair at mouse moves
        self.crosshair_update = SignalProxy(
            self.plot_graph.scene().sigMouseMoved, rateLimit=60,
            slot=self.update_crosshair)

        # Set the graph title
        self.plot_graph.setTitle("<span style='color: #FFAE42; "
                                 "font-size: 11pt'>dose/fx: %0.2f</span>, "
                                 "<span style='color: #FFAE42; "
                                 "font-size: 11pt'>vRel: %0.1f</span>"
                                 % (0, 0.0))

        # 
        self.plot_graph.plotItem.vb.setLimits(
            xMin=0, xMax=max(dose_histogram['evaluation_points']),
            yMin=-0.1, yMax=100.1)

        # 
        self.plot_graph.plotItem.vb.enableAutoRange()

    def get_segment_statistics(self, event):
        """."""

        # 
        dosimetrics = (self.parent.plans[self.parent.plan_ledit.text()]
                       .datahub.dosimetrics)

        # 
        self.parent.segment_ledit.setText(event.name())

        # 
        self.parent.mean_ledit.setText(str(
            round(dosimetrics[event.name()]['mean'], 2)))

        # 
        self.parent.std_ledit.setText(str(
            round(dosimetrics[event.name()]['std'], 2)))

        # 
        self.parent.maximum_ledit.setText(str(
            round(dosimetrics[event.name()]['max'], 2)))

        # 
        self.parent.minimum_ledit.setText(str(
            round(dosimetrics[event.name()]['min'], 2)))

    def reset_dvh(self):
        """."""

        self.plot_graph.clear()
        self.plot_graph.getPlotItem().hideAxis('bottom')
        self.plot_graph.getPlotItem().hideAxis('top')
        self.plot_graph.getPlotItem().hideAxis('left')
        self.plot_graph.getPlotItem().hideAxis('right')
        self.plot_graph.setTitle(None)
        if hasattr(self, 'crosshair_update'):
            delattr(self, 'crosshair_update')
        self.parent.segment_ledit.clear()
        self.parent.mean_ledit.clear()
        self.parent.std_ledit.clear()
        self.parent.maximum_ledit.clear()
        self.parent.minimum_ledit.clear()

    def select_dvh_curve(self, event):
        """."""

        # Get all plot items
        items = self.plot_graph.getPlotItem().listDataItems()

        for item in items:
            pen = item.curve.opts['pen']
            if item == event:
                pen = mkPen(pen)
                if pen.width() == 2:
                    event.curve.setPen(mkPen(color=pen.color(),
                                             style=pen.style(),
                                             width=5))
                else:
                    event.curve.setPen(mkPen(color=pen.color(),
                                             style=pen.style(),
                                             width=2))
                    self.parent.segment_ledit.clear()
                    self.parent.mean_ledit.clear()
                    self.parent.std_ledit.clear()
                    self.parent.maximum_ledit.clear()
                    self.parent.minimum_ledit.clear()
            else:
                item.curve.setPen(mkPen(color=pen.color(),
                                        style=pen.style(),
                                        width=2))

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

            if 0 <= mouse_point.x() and 0 <= mouse_point.y() <= 100:

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

    def update_dvh(self):
        """."""

        for segment in self.segments:

            pen = mkPen(color=self.segment_styles[segment][0],
                        style=self.segment_styles[segment][1],
                        width=2)

            plot = self.plot_graph.plot(
                self.dose_histogram['evaluation_points'],
                self.dose_histogram[segment]['dvh_values'],
                pen=pen, name=segment, clickable=True)
            plot.sigClicked.connect(self.get_segment_statistics)
            plot.sigClicked.connect(self.select_dvh_curve)
