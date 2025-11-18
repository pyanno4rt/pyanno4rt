"""Outcome graph widget."""

# Author: Tim Ortkamp

# %% External package import

from itertools import islice, cycle
from numpy import multiply
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QVBoxLayout, QWidget
from pyqtgraph import colormap, mkPen, PlotWidget, setConfigOptions

# %% Internal package import

from pyanno4rt.visualization._custom_styles import tooltip

# %% Plotting options

setConfigOptions(antialias=True)

# %% Class definition


class OutcomeGraphWidget(QWidget):
    """
    Outcome graph widget for the visualizer.

    This class sets up an outcome graph widget for the visual analysis tool, \
    including a line plot with the iterative outcome predictions.

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

        # Initialize the component tracker and the display styles
        self.tracker = None
        self.styles = None

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

        # Get the number of tracks
        track_num = len(tracker)

        # Set the marker styles
        markers = tuple(islice(
            cycle(['o', 's', 't', 'd', 'star', 'x']), track_num))

        # Set the colormap
        colors = colormap.get(
            'tab20b', 'matplotlib').getLookupTable(nPts=track_num)

        # Set the line styles
        linestyles = tuple(islice(
            cycle([Qt.SolidLine, Qt.DashLine, Qt.DotLine, Qt.DashDotLine]),
            track_num))

        # Create a dictionary for the track styles
        self.styles = dict(
            zip(self.tracker, tuple(zip(markers, colors, linestyles))))

        # Set the plot labels
        self.plot_widget.setLabels(
            left="Outcome prediction [%]", bottom="Evaluation step",
            right=" ", top=" ")

        # Set the plot limits
        self.plot_widget.plotItem.vb.setLimits(
            xMin=0, xMax=max(len(track) for track in tracker.values())+1,
            yMin=-5, yMax=105)

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
            if item.curve == event:

                # Construct the QPen
                pen = mkPen(pen)

                # Check if the pen width is 2
                if pen.width() == 2:

                    # Increase the symbol size
                    item.setSymbolSize(14)

                    # Increase the pen width
                    item.curve.setPen(mkPen(
                        color=pen.color(), style=pen.style(), width=4))

                    # Get the optimization data
                    optimization = self.parent.plan.datahub.optimization

                    # Get the optimization components
                    components = (
                        optimization['problem'].objectives
                        | optimization['problem'].constraints)

                    # Get the selected instance attribute
                    component_type, embedding, weight, rank, bounds = (getattr(
                        components[item.name()]['instance'], attribute)
                        for attribute in (
                            'component_type', 'embedding', 'weight', 'rank',
                            'bounds'))

                    # Display the component name
                    self.parent.outc_component_ledit.setText(item.name())

                    # Display the component type
                    self.parent.outc_type_ledit.setText(component_type)

                    # Display the component embedding mode
                    self.parent.outc_embedding_ledit.setText(embedding)

                    # Display the component weight
                    self.parent.outc_weight_ledit.setText(str(weight))

                    # Display the component rank
                    self.parent.outc_rank_ledit.setText(str(rank))

                    # Display the component bounds
                    self.parent.outc_bounds_ledit.setText(str(bounds))

                else:

                    # Decrease the symbol size
                    item.setSymbolSize(7)

                    # Decrease the pen width
                    item.curve.setPen(mkPen(
                        color=pen.color(), style=pen.style(), width=2))

                    # Clear the component line editors
                    self.parent.outc_component_ledit.clear()
                    self.parent.outc_type_ledit.clear()
                    self.parent.outc_embedding_ledit.clear()
                    self.parent.outc_weight_ledit.clear()
                    self.parent.outc_rank_ledit.clear()
                    self.parent.outc_bounds_ledit.clear()

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
            self.parent.outc_component_ledit.clear()
            self.parent.outc_type_ledit.clear()
            self.parent.outc_embedding_ledit.clear()
            self.parent.outc_weight_ledit.clear()
            self.parent.outc_rank_ledit.clear()
            self.parent.outc_bounds_ledit.clear()

    def update_graph(self):
        """Update the outcome graph."""

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
                range(1, len(self.tracker[track])+1),
                multiply(self.tracker[track], 100),
                pen=pen,
                symbol=self.styles[track][0],
                symbolSize=7,
                symbolBrush=[self.styles[track][1] for i in range(track_len)],
                name=track,
                clickable=True)

            # Add hoverable tooltip to the scatter points
            plot.scatter.opts.update(
                hoverable=True,
                tip='step: {x:0.0f}\nprediction: {y:0.4f} %'.format)

            # Connect the plot with the curve selection methods
            plot.sigClicked.connect(self.select_curve)
            self.plot_widget.scene().sigMouseClicked.connect(
                self.unselect_curve)

    def reset_graph(self):
        """Reset the outcome graph."""

        # Clear the plot graph
        self.plot_widget.clear()

        # Hide the axes
        self.plot_widget.getPlotItem().hideAxis('bottom')
        self.plot_widget.getPlotItem().hideAxis('top')
        self.plot_widget.getPlotItem().hideAxis('left')
        self.plot_widget.getPlotItem().hideAxis('right')

        # Clear the component line editors
        self.parent.outc_component_ledit.clear()
        self.parent.outc_type_ledit.clear()
        self.parent.outc_embedding_ledit.clear()
        self.parent.outc_weight_ledit.clear()
        self.parent.outc_rank_ledit.clear()
        self.parent.outc_bounds_ledit.clear()
