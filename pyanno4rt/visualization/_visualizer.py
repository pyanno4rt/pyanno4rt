"""Visualizer."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QHBoxLayout, QLabel, QMainWindow, QPushButton,
                             QSizePolicy, QVBoxLayout, QWidget)
from pyqtgraph import mkQApp, setConfigOptions
from pyqtgraph.Qt import QtGui

# %% Internal package import

from pyanno4rt.datahub import Datahub
from pyanno4rt.tools import get_model_objectives, get_objectives
from pyanno4rt.visualization.visuals import (
    CtDoseSlicingWindowPyQt, DosimetricsTablePlotterMPL, DVHGraphPlotterMPL,
    FeatureSelectWindowPyQt, IterGraphPlotterMPL, MetricsGraphsPlotterMPL,
    MetricsTablesPlotterMPL, NTCPGraphPlotterMPL,
    PermutationImportancePlotterMPL)

# %% Set options

setConfigOptions(imageAxisOrder='col-major')

# %% Class definition


class Visualizer():
    """
    Visualizer class.

    This class provides methods to build and launch the visual analysis tool, \
    i.e., it initializes the application, creates the main window, provides \
    the window configuration, and runs the application.

    Attributes
    ----------
    application : object of class `SpyderQApplication`
        Instance of the class `SpyderQApplication` for managing control flow \
        and main settings of the visual analysis tool.
    """

    def __init__(self, parent=None):

        # Log a message about the initialization of the class
        Datahub().logger.display_info("Initializing visualizer class ...")

        if not parent:

            # Initialize the default parent
            self.parent = None

            # Initialize the application
            self.application = mkQApp("Visualizer")

            # Set the application style
            self.application.setStyle('Fusion')

            # Initialize the main window for the GUI
            self.main_window = MainWindow(self.application)

        else:

            # Initialize the parent
            self.parent = parent

            # Initialize the main window for the GUI from the parent
            self.parent.visual_window = MainWindow(
                self.parent.application, False)

    def launch(self):
        """Launch the visual analysis tool."""
        # Log a message about the analysis tool launching
        Datahub().logger.display_info("Launching visual analysis tool ...")

        if not self.parent:

            # Set the window size
            self.main_window.resize(1000, 300)

            # Show the main window
            self.main_window.show()

            # Run the application
            self.application.exec_()

        else:

            # Set the window size
            self.parent.visual_window.resize(
                int(0.7*self.parent.width()), 300)

            # Get the window geometry
            geometry = self.parent.visual_window.geometry()

            # Move the geometry center towards the parent
            geometry.moveCenter(self.parent.geometry().center())

            # Set the shifted geometry
            self.parent.visual_window.setGeometry(geometry)

            # Show the visualization window
            self.parent.visual_window.show()


class MainWindow(QMainWindow):
    """
    Main window for the application.

    This class creates the main window for the visual analysis tool, \
    including logo, labels, and event buttons.

    Parameters
    ----------
    application : object of class `SpyderQApplication`
        Instance of the class `SpyderQApplication` for managing control flow \
        and main settings of the visual analysis tool.
    """

    def __init__(
            self,
            application,
            standalone=True):

        # Initialize the datahub
        hub = Datahub()

        def add_logo(layout):
            """Create and add the pyanno4rt logo."""
            logo = QLabel(self)
            pixmap = QtGui.QPixmap('./logo/logo_white.png')
            pixmap = pixmap.scaled(int(pixmap.width()/10),
                                   int(pixmap.height()/10))
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

        def add_label(layout):
            """Create and add the label below the logo."""
            label = QLabel("Visual Analysis Tool")
            label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet('''
                                QLabel
                                    {
                                        color: #FBFAF5;
                                        font-size: 10pt;
                                        margin-bottom: 15px;
                                    }
                                ''')
            layout.addWidget(label)

        def add_button(subclass, layouts, counts):
            """Create a button and add it to the respective layout."""
            # Get the category index
            indexes = {labels[0]: 0, labels[1]: 1, labels[2]: 2}
            index = indexes[subclass.category]

            # Create a push button
            button = QPushButton(subclass.label)

            # Set the pointing hand cursor for the button
            button.setCursor(QtGui.QCursor(Qt.PointingHandCursor))

            # Set the stylesheet for the button
            button.setStyleSheet(button_styles[index])

            # Add the button to the layout
            layouts[index].addWidget(button)

            # Increment the counter for the category
            counts[index] += 1

            # Connect the buttons with the onclick events
            button.clicked.connect(getattr(self, subclass.name).view)

            # Disable the iteration value buttons if no tracker is present
            # or if no objective should be displayed
            if ((not hasattr(
                    hub.optimization['problem'], 'tracker')
                    or not any(
                        objective.display for objective in objectives))
                    and subclass.name == 'iterations_plotter'):
                button.setEnabled(False)

            # Disable the iteration value buttons if no tracker is present
            # or if no data objective should be displayed
            if ((not hasattr(
                    hub.optimization['problem'], 'tracker')
                    or not any(
                        objective.display for objective in data_objectives))
                    and subclass.name == 'ntcp_plotter'):
                button.setEnabled(False)

            # Disable the data model buttons if no models are present
            if (len(data_objectives) == 0
                    and hasattr(subclass, 'DATA_DEPENDENT')):
                button.setEnabled(False)

            # Disable the features button if no features history exists
            if (len(data_objectives) > 0 and
                    all(objective.model_parameters['write_features'] is False
                        for objective in data_objectives)
                    and subclass.name == 'features_plotter'):
                button.setEnabled(False)

            # Disable the permutation importance button if no inspector is used
            if (len(hub.model_inspections) == 0 and subclass.name in (
                    'permutation_importance_plotter',)):
                button.setEnabled(False)

            # Disable the metrics tables and graphs if no evaluator is used
            if (len(hub.model_evaluations) == 0 and subclass.name in (
                    'metrics_graphs_plotter', 'metrics_tables_plotter')):
                button.setEnabled(False)

        # Run the constructor from the superclass
        super().__init__()

        # Get the instance attributes from the arguments
        self.application = application
        self.standalone = standalone

        # Set the window title
        self.setWindowTitle("pyanno4rt - visual analysis tool")

        # Set the window style sheet
        self.setStyleSheet('background-color: black;')

        # Initialize the central widget and add it to the window
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        # Set the layout for the central widget
        central_layout = QVBoxLayout()
        central_widget.setLayout(central_layout)

        # Add logo and label to the central layout
        add_logo(central_layout)
        add_label(central_layout)

        # Set the horizontal box layout for the plotting categories
        categories_layout = QHBoxLayout()

        # Add the categories layout to the central layout
        central_layout.addLayout(categories_layout)

        # Set the stylesheets for the category labels
        label_styles = ('''
                        QLabel
                            {
                                color: #C45C26;
                                font-size: 14pt;
                                max-height: 25%;
                            }
                        ''',
                        '''
                        QLabel
                            {
                                color: #34A56F;
                                font-size: 14pt;
                                max-height: 25%;
                            }
                        ''',
                        '''
                        QLabel
                            {
                                color: #5CB3FF;
                                font-size: 14pt;
                                max-height: 25%;
                            }
                        ''')

        # Set the stylesheet for the category buttons
        button_styles = ('''
                         QPushButton
                             {
                                 background-color: #C45C26;
                             }
                         QPushButton:hover
                             {
                                 background-color: #B04812;
                             }
                         QPushButton:pressed
                             {
                                 background-color: #B04812;
                             }
                         QPushButton:disabled
                             {
                                 color: #808080;
                             }
                         ''',
                         '''
                         QPushButton
                             {
                                 background-color: #34A56F;
                             }
                         QPushButton:hover
                             {
                                 background-color: #278664;
                             }
                         QPushButton:pressed
                             {
                                 background-color: #278664;
                             }
                         QPushButton:disabled
                             {
                                 color: #808080;
                             }
                         ''',
                         '''
                         QPushButton
                             {
                                 background-color: #5CB3FF;
                             }
                         QPushButton:hover
                             {
                                 background-color: #157DEC;
                             }
                         QPushButton:pressed
                             {
                                 background-color: #157DEC;
                             }
                         QPushButton:disabled
                             {
                                 color: #808080;
                             }
                         ''')

        # Initialize all plotting classes
        classes = (IterGraphPlotterMPL,
                   NTCPGraphPlotterMPL,
                   FeatureSelectWindowPyQt,
                   MetricsGraphsPlotterMPL,
                   MetricsTablesPlotterMPL,
                   PermutationImportancePlotterMPL,
                   DVHGraphPlotterMPL,
                   DosimetricsTablePlotterMPL,
                   CtDoseSlicingWindowPyQt)

        # Set the category labels
        labels = ('Optimization problem analysis',
                  'Data-driven model review',
                  'Treatment plan evaluation')

        # Create the layouts for all categories
        layouts = []
        for i, _ in enumerate(labels):
            layouts.append(QVBoxLayout())
            text = QLabel(labels[i])
            text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            text.setAlignment(Qt.AlignCenter)
            text.setStyleSheet(label_styles[i])
            layouts[i].addWidget(text)

        # Get all objectives and the data-dependent objectives
        objectives = get_objectives(hub.segmentation)
        data_objectives = get_model_objectives(hub.segmentation)

        # Initialize the counter
        counts = [0, 0, 0]

        # Loop over the plotting classes
        for subclass in classes:

            # Set the class attribute from the subclasses' name
            setattr(self, subclass.name, subclass())

            # Add the button to the respective category layout
            add_button(subclass, layouts, counts)

        # Loop over the vertical button layouts
        for layout in layouts:

            # Add the layout to the categories layout
            categories_layout.addLayout(layout)

        # Get the maximum number of plots over all categories
        equal_number = max(counts)

        # Get the number of spacers to add per category
        additional_space = {str(n): equal_number - counts[n]
                            for n, _ in enumerate(counts)}

        # Loop over the items
        for key, value in additional_space.items():

            # Loop over the value ranges
            for _ in range(value):

                # Create and add a spacer button
                spacer_button = QPushButton("")
                spacer_button.setStyleSheet('''
                                            QPushButton
                                                {
                                                    color: black;
                                                    background: black;
                                                    border-color: black;
                                                }
                                            ''')
                layouts[int(key)].addWidget(spacer_button)

    def closeEvent(
            self,
            event):
        """
        Close the application.

        Parameters
        ----------
        event : object of class `QCloseEvent`
            Instance of the class `QCloseEvent`.
        """
        # Log a message about the analysis tool closing
        Datahub().logger.display_info("Closing visual analysis tool ...")

        # Check if the visual interface is handled as a standalone
        if self.standalone:

            # Close the application
            self.application.quit()

        else:

            # Hide the window
            self.hide()
