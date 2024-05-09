"""Feature selection window (PyQt)."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from numpy import array, ceil

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QComboBox, QHBoxLayout, QLabel, QMainWindow,
                             QSizePolicy, QVBoxLayout, QWidget)
from pyqtgraph import (mkPen, PlotWidget, setConfigOptions, TableWidget,
                       InfiniteLine, SignalProxy)
from pyqtgraph.Qt import QtGui

# %% Internal package import

from pyanno4rt.datahub import Datahub
from pyanno4rt.tools import (
    get_machine_learning_constraints, get_machine_learning_objectives)

# %% Set options

setConfigOptions(imageAxisOrder='col-major')

# %% Class definition


class ComboBoxModel(QWidget):
    """
    Combo box for model selection.

    This class provides a combo box for picking the model to be displayed \
    in the feature selection window.

    Parameters
    ----------
    models : tuple
        Tuple with the model names to display in the combo box.

    Attributes
    ----------
    model_box : object of class `QComboBox`
        Instance of the class `QComboBox`, which creates a combo box to \
        select the model to be displayed.
    """

    def __init__(
            self,
            model_names):

        # Call the superclass constructor
        super().__init__()

        # Set the vertical layout for the widget
        vertical_layout = QVBoxLayout(self)

        # Set the vertical layout for the combo box
        box_layout = QVBoxLayout()

        # Create the combo box
        self.model_box = QComboBox()

        # Set the pointing hand cursor for the closed/open feature box
        self.model_box.setCursor(QtGui.QCursor(Qt.PointingHandCursor))
        self.model_box.view().setCursor(QtGui.QCursor(Qt.PointingHandCursor))

        # Add the feature names to the feature box
        self.model_box.addItems(model_names)

        # Set the stylesheet for the feature box
        self.model_box.setStyleSheet('''
                                     QComboBox
                                         {
                                             background-color: black;
                                             color: #FBFAF5;
                                         }
                                     QComboBox:hover
                                         {
                                             background-color: #B04812;
                                             color: black;
                                         }
                                     QComboBox::item
                                         {
                                             color: #FBFAF5;
                                         }
                                     QComboBox::item:selected
                                         {
                                             background-color: #FFAE42;
                                             color: black;
                                         }
                                     QComboBox::item:checked
                                         {
                                             background-color: #C45C26;
                                             color: black;
                                         }
                                     QComboBox:!on
                                         {
                                             background-color: #C45C26;
                                             color: black;
                                             padding-left: 5%;
                                             padding-right: 5%;
                                             margin-left: 5%;
                                             margin-right: 5%;
                                         }
                                     ''')

        # Add the feature box to the box layout
        box_layout.addWidget(self.model_box)

        # Add the box layout to the vertical layout
        vertical_layout.addLayout(box_layout)

        # Resize the combo box
        self.resize(self.sizeHint())


class ComboBoxFeature(QWidget):
    """
    Combo box for feature selection.

    This class provides a combo box for picking the feature to be displayed \
    in the feature selection window.

    Parameters
    ----------
    feature names : tuple
        Tuple with the feature names to display in the combo box.

    Attributes
    ----------
    feature_box : object of class `QComboBox`
        Instance of the class `QComboBox`, which creates a combo box to \
        select the feature to be displayed.
    """

    def __init__(
            self,
            feature_names):

        # Call the superclass constructor
        super().__init__()

        # Set the vertical layout for the widget
        vertical_layout = QVBoxLayout(self)

        # Set the vertical layout for the combo box
        box_layout = QVBoxLayout()

        # Create the combo box
        self.feature_box = QComboBox()

        # Set the pointing hand cursor for the closed/open feature box
        self.feature_box.setCursor(QtGui.QCursor(Qt.PointingHandCursor))
        self.feature_box.view().setCursor(QtGui.QCursor(Qt.PointingHandCursor))

        # Add the feature names to the feature box
        self.feature_box.addItems(feature_names)

        # Set the stylesheet for the feature box
        self.feature_box.setStyleSheet('''
                                       QComboBox
                                           {
                                               background-color: black;
                                               color: #FBFAF5;
                                           }
                                       QComboBox:hover
                                           {
                                               background-color: #B04812;
                                               color: black;
                                           }
                                       QComboBox::item
                                           {
                                               color: #FBFAF5;
                                           }
                                       QComboBox::item:selected
                                           {
                                               background-color: #FFAE42;
                                               color: black;
                                           }
                                       QComboBox::item:checked
                                           {
                                               background-color: #C45C26;
                                               color: black;
                                           }
                                       QComboBox:!on
                                           {
                                               background-color: #C45C26;
                                               color: black;
                                               padding-left: 5%;
                                               padding-right: 5%;
                                               margin-left: 5%;
                                               margin-right: 5%;
                                           }
                                       ''')

        # Add the feature box to the box layout
        box_layout.addWidget(self.feature_box)

        # Add the box layout to the vertical layout
        vertical_layout.addLayout(box_layout)

        # Resize the combo box
        self.resize(self.sizeHint())


class FeatureWidget(QWidget):
    """
    Feature widget class.

    This class provides a feature widget, including a combo box (to select \
    the feature to be displayed), a plot widget (to show the iterative \
    feature values), and a table widget (to list the iterative feature values).

    Parameters
    ----------
    feature_history : dict
        Dictionary with the iterative feature values.

    Attributes
    ----------
    label : object of class `QLabel`
        Instance of the class `QLabel`, which holds the label to be displayed \
        above the graph.

    feature_history : dict
        See `Parameters`.

    combo_box_model : object of class `ComboBoxModel`
        Instance of the class `ComboBoxModel`, which generates the combo box \
        for the selection of the model to be displayed.

    combo_box_feature : object of class `ComboBoxFeature`
        Instance of the class `ComboBoxFeature`, which generates the combo \
        box for the selection of the feature to be displayed.

    graph : object of class `PlotWidget`
        Instance of the class `PlotWidget`, which generates the graph for the \
        feature values.

    table : object of class `TableWidget`
        Instance of the class `TableWidget`, which generates the table for \
        the feature values.

    evaluation_steps : ndarray
        Array with the evaluation steps as integers (starting with 1).

    feature_values : ndarray
        Array with the values of the selected feature for all evaluation steps.

    pen : object of class `QPen`
        Instance of the class `QPen`, which defines how to draw lines and \
        outlines of shapes.

    vertical_line : object of class `InfiniteLine`
        Instance of the class `InfiniteLine`, which represents an infinite \
        vertical line to indicate the x-axis position in the graph.

    horizontal_line : object of class `InfiniteLine`
        Instance of the class `InfiniteLine`, which represents an infinite \
        horizontal line to indicate the y-axis position in the graph.

    crosshair_update : object of class `SignalProxy`
        Instance of the class `SignalProxy`, which connects mouse moving with \
        the update of the crosshair.
    """

    def __init__(
            self,
            feature_histories):

        # Call the superclass constructor
        super().__init__()

        def add_label(layout, label):
            """Create and add the label above the graph."""
            self.label = QLabel(label)
            self.label.setStyleSheet('''
                                     QLabel
                                         {
                                             color: #FBFAF5;
                                             font-size: 12pt;
                                             max-height: 25%;
                                             margin-bottom: 10px;
                                         }
                                     ''')
            self.label.setSizePolicy(QSizePolicy.Expanding,
                                     QSizePolicy.Expanding)
            self.label.setAlignment(Qt.AlignCenter)
            layout.addWidget(self.label)

        # Get the feature histories from the argument
        self.feature_histories = feature_histories

        # Set the horizontal layout for the feature widget
        feature_layout = QHBoxLayout(self)

        # Set the vertical layout for the combo boxes
        boxes_layout = QVBoxLayout()
        boxes_layout.addStretch()

        # Initialize the model combo box and add it to the feature layout
        self.combo_box_model = ComboBoxModel((*self.feature_histories.keys(),))
        boxes_layout.addWidget(self.combo_box_model)

        # Initialize the feature combo box and add it to the feature layout
        self.combo_box_feature = ComboBoxFeature(
            (*self.feature_histories[
                self.combo_box_model.model_box.currentText()].keys(),))
        boxes_layout.addWidget(self.combo_box_feature)

        # Add vertical stretch to the boxes layout
        boxes_layout.addStretch()

        # Add the boxes layout to the feature layout
        feature_layout.addLayout(boxes_layout)

        # Set the vertical layout for the graph
        graph_layout = QVBoxLayout()

        # Add the label to the graph layout
        add_label(graph_layout,
                  self.combo_box_feature.feature_box.currentText())

        # Initialize the plot widget
        self.graph = PlotWidget()

        # Set the axis labels of the graph
        self.graph.setLabels(left="Value", bottom="Evaluation step")

        # Add the graph to the graph layout
        graph_layout.addWidget(self.graph)

        # Add the graph layout to the feature layout
        feature_layout.addLayout(graph_layout)

        # Set the vertical layout for the table
        table_layout = QVBoxLayout()

        # Initialize the table widget and show it
        self.table = TableWidget()
        self.table.show()

        # Set the stylesheet for the table
        self.table.setStyleSheet('''
                                 TableWidget
                                     {
                                         background-color: #FFAE42;
                                         color: black;
                                         border: 1px solid white;
                                         selection-background-color: #C45C26;
                                         max-height: 450%;
                                         margin-left: 40px;
                                     }
                                 QHeaderView::section
                                     {
                                         background: #C45C26;
                                     }
                                 QScrollBar
                                     {
                                         background-color: #FBFAF5;
                                         color: black;
                                     }
                                 QScrollBar::sub-page
                                     {
                                         background: black;
                                         border: 1px solid #FBFAF5;
                                     }
                                 QScrollBar::add-page
                                     {
                                         background: black;
                                         border: 1px solid #FBFAF5;
                                     }
                                ''')

        # Add the table to the table layout
        table_layout.addWidget(self.table)

        # Add the table layout to the feature layout
        feature_layout.addLayout(table_layout)

        # Connect the model combo box and the plot/table
        self.combo_box_model.model_box.currentTextChanged.connect(
            self.update_plot)
        self.combo_box_model.model_box.currentTextChanged.connect(
            self.update_table)

        # Connect the feature combo box and the plot/table
        self.combo_box_feature.feature_box.currentTextChanged.connect(
            self.update_plot)
        self.combo_box_feature.feature_box.currentTextChanged.connect(
            self.update_table)

        # Connect the model box with the feature box
        self.combo_box_model.model_box.currentIndexChanged.connect(
            self.update_combo_box_feature)

        # Connect the table with marking points
        self.table.cellClicked.connect(self.mark_points)

        # Update plot and table initially
        self.update_plot()
        self.update_table()

    def update_combo_box_feature(self):
        """Update the feature combo box for model selection changes."""
        # Block signals from the feature combo box
        self.combo_box_feature.feature_box.blockSignals(True)

        # Clear the current feature combo box content
        self.combo_box_feature.feature_box.clear()

        # Add the new items to the feature combo box
        self.combo_box_feature.feature_box.addItems(
            self.feature_histories[
                self.combo_box_model.model_box.currentText()].keys())

        # Re-enable signals from the feature combox box
        self.combo_box_feature.feature_box.blockSignals(False)

    def update_plot(self):
        """Update the plot for feature selection changes."""
        # Get the current feature history
        feature_history = self.feature_histories[
                        self.combo_box_model.model_box.currentText()]

        # Get the selected feature label
        selected_feature = self.combo_box_feature.feature_box.currentText()

        # Overwrite the plot label
        self.label.setText(selected_feature)

        # Remove all items from the graph
        self.graph.clear()

        # Get the plot item
        plot_item = self.graph.getPlotItem()

        # Get the bottom plot axis
        bottom_axis = plot_item.getAxis('bottom')

        # Set the tick spacing for the x-axis
        steps_in_x = min(
            (5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000),
            key=lambda x: abs(ceil(
                max(len(history) for history
                    in feature_history.values())/x)
                - 20))
        x_ticks = [steps_in_x, steps_in_x/5]
        bottom_axis.setTickSpacing(x_ticks[0], x_ticks[1])

        # Get the left plot axis
        left_axis = plot_item.getAxis('left')

        # Determine the y-ticks from the feature values
        y_ticks = [(max(feature_history[selected_feature])
                    - min(feature_history[selected_feature]))
                   / 10,
                   (max(feature_history[selected_feature])
                    - min(feature_history[selected_feature]))
                   / 50]

        # Check if the feature has constant values
        if all(ticks == 0 for ticks in y_ticks):

            # Replace the tick spacing for the y-axis with nonzero values
            y_ticks = [
                2*max(feature_history[selected_feature]),
                0.4*max(feature_history[selected_feature])]

            # Check if the ticks are still zero
            if all(tick == 0 for tick in y_ticks):

                # Add a default value to the ticks
                y_ticks = [10, 2]

        # Set the tick spacing for the y-axis
        left_axis.setTickSpacing(y_ticks[0], y_ticks[1])

        # Show the grid for both axes
        plot_item.showGrid(x=True, y=True, alpha=0.3)

        # Get the axis values from the history
        self.evaluation_steps = array(
            range(1, len(feature_history[selected_feature])+1))
        self.feature_values = feature_history[selected_feature]

        # Initialize the QPen with a color
        self.pen = mkPen(color='#5CB3FF')

        # Plot the feature values over the evaluation steps
        self.graph.plot(self.evaluation_steps, self.feature_values,
                        pen=self.pen, symbol='o', symbolSize=7,
                        symbolBrush=('#FFAE42'))

        # Set the graph title
        self.graph.setTitle("<span style='color: #FFAE42; "
                            "font-size: 8pt'>evaluation step: %0i</span>, "
                            "<span style='color: #FFAE42; "
                            "font-size: 8pt'>value: %0.1f</span>"
                            % (0, 0.0))

        # Create vertical and horizontal infinite lines
        self.vertical_line = InfiniteLine(angle=90)
        self.horizontal_line = InfiniteLine(angle=0, movable=False)

        # Disable the pens for the lines
        self.vertical_line.setPen(None)
        self.horizontal_line.setPen(None)

        # Add the lines to the graph
        self.graph.addItem(self.vertical_line, ignoreBounds=True)
        self.graph.addItem(self.horizontal_line, ignoreBounds=True)

        # Set the signal proxy to update the crosshair at mouse moves
        self.crosshair_update = SignalProxy(self.graph.scene().sigMouseMoved,
                                            rateLimit=60,
                                            slot=self.update_crosshair)

    def update_table(self):
        """Update the table for feature selection changes."""
        # Get the current feature history
        feature_history = self.feature_histories[
                        self.combo_box_model.model_box.currentText()]

        # Set the data for the table
        self.table.setData(feature_history[
            self.combo_box_feature.feature_box.currentText()].round(5))

        # Set the table header
        self.table.setHorizontalHeaderLabels(['Value'])

        # Resize the table cells to the contents
        self.table.resizeColumnsToContents()
        self.table.resizeRowsToContents()

    def mark_points(self):
        """Mark the graph points when clicking on table cells."""
        # Get the current feature history
        feature_history = self.feature_histories[
                        self.combo_box_model.model_box.currentText()]

        # Remove all items from the graph
        self.graph.clear()

        # Plot the feature values over the evaluation steps
        self.graph.plot(self.evaluation_steps, self.feature_values,
                        pen=self.pen, symbol='o', symbolSize=7,
                        symbolBrush=('#FFAE42'))

        # Get the indices from the selected table cells
        indices = tuple(
            int(index.row())+1 for index in self.table.selectedIndexes())

        # Get the corresponding values from the feature history
        values = tuple(feature_history[
            self.combo_box_feature.feature_box.currentText()][index-1]
            for index in indices)

        # Overlay the plot points with a red circle
        self.graph.plot(indices, values, symbol='o', symbolSize=14,
                        symbolBrush=('#C45C26'))

    def update_crosshair(
            self,
            event):
        """Update the crosshair at mouse moves."""
        # Get the coordinates from the triggered event
        coordinates = event[0]

        # Check if the coordinates lie within the scene bounding rectangle
        if self.graph.sceneBoundingRect().contains(coordinates):

            # Get the mouse point in the view's coordinate system
            mouse_point = self.graph.plotItem.vb.mapSceneToView(coordinates)

            # Get the index from the mouse point
            index = mouse_point.x()

            # Check if the index lies in the evaluation steps interval
            if self.evaluation_steps[0] < index <= self.evaluation_steps[-1]+1:

                # Update the graph title
                self.graph.setTitle("<span style='color: #FFAE42; "
                                    "font-size: 8pt'>evaluation step: "
                                    "%0i</span>, <span style='color: #FFAE42; "
                                    "font-size: 8pt'>value: %0.1f</span>"
                                    % (int(round(mouse_point.x())),
                                       mouse_point.y()))

            # Update the positions of vertical and horizontal lines
            self.vertical_line.setPos(mouse_point.x())
            self.horizontal_line.setPos(mouse_point.y())


class FeatureSelectWindowPyQt(QMainWindow):
    """
    Feature selection window (PyQt) class.

    This class provides an interactive plot of the iterative feature values, \
    including a combo box for feature selection, a graph plot with the value \
    per iteration, and a value table as a second representation.

    Attributes
    ----------
    DATA_DEPENDENT : bool
        Indicator for the assignment to model-related plots.

    category : string
        Plot category for assignment to the button groups in the visual \
        interface.

    name : string
        Attribute name of the classes' instance in the visual interface.

    label : string
        Label of the plot button in the visual interface.
    """

    # Set the flag for the model-related plots
    DATA_DEPENDENT = True

    # Set the class attributes for the visual interface integration
    category = "Optimization problem analysis"
    name = "features_plotter"
    label = "Iterative feature calculation plot"

    def view(self):
        """Open the full-screen view on the feature selection window."""
        # Initialize the datahub
        hub = Datahub()

        # Log a message about the plot opening
        hub.logger.display_info("Opening iterative feature calculation plot "
                                "...")

        # Get the feature histories from the feature calculators
        self.feature_histories = {
            component.model.model_label:
                component.data_model_handler.feature_calculator.feature_history
            for component in (
                    get_machine_learning_constraints(hub.segmentation)
                    + get_machine_learning_objectives(hub.segmentation))
            if hasattr(component.data_model_handler.feature_calculator,
                       'feature_history')}

        def add_logo(layout):
            """Create and add the pyanno4rt logo."""
            logo = QLabel(self)
            pixmap = QtGui.QPixmap('./logo/logo_white_512.png')
            pixmap = pixmap.scaled(int(pixmap.width()/2),
                                   int(pixmap.height()/2))
            logo.setPixmap(pixmap)
            logo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            logo.setAlignment(Qt.AlignCenter)
            logo.setStyleSheet('''
                               QLabel
                                   {
                                       margin: 0 0 0 0;
                                   }
                               ''')
            layout.addWidget(logo)

        # Set the window title
        self.setWindowTitle("pyanno4rt - feature graphs")

        # Set the window style sheet
        self.setStyleSheet('background-color: black;')

        # Initialize the central widget and add it to the window
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        # Set the vertical layout for the central widget
        central_layout = QVBoxLayout()
        central_widget.setLayout(central_layout)

        # Add the logo to the central layout
        add_logo(central_layout)

        # Create the feature widget and add it to the layout
        feature_widget = FeatureWidget(self.feature_histories)
        central_layout.addWidget(feature_widget)

        # Show the plot in screen size
        self.showMaximized()
