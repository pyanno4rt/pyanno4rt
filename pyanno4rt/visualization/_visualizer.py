"""Visualizer."""

# Author: Tim Ortkamp

# %% External package import

from pyqtgraph import mkQApp
from PyQt5.QtWidgets import QApplication, QMainWindow

# %% Internal package import

from pyanno4rt.datahub import Datahub
from pyanno4rt.tools import (
    get_conventional_constraints, get_conventional_objectives,
    get_machine_learning_constraints, get_machine_learning_objectives,
    get_radiobiological_constraints, get_radiobiological_objectives)
from pyanno4rt.visualization._custom_styles import pbutton_composer
from pyanno4rt.visualization.design.visualizer import Ui_vis_window
from pyanno4rt.visualization.static import (
    DosimetricsTable, DVHGraph, IterGraph, MetricsGraph, MetricsTable,
    NTCPGraph, PermutationImportanceBoxplot)

# %% Class definition


class Visualizer(QMainWindow, Ui_vis_window):
    """
    Visualizer class.

    This class creates the visual analysis tool as standalone or for the \
    graphical user interface, including different types of plots.
    """

    def __init__(
            self,
            parent=None):

        # Run the constructor from the superclass
        super().__init__()

        # Build the UI main window
        self.setupUi(self)

        # Initialize the application
        self.application = mkQApp("pyanno4rt")

        # Set the application style
        self.application.setStyle('Fusion')

        # Get the application from the argument
        self.parent = parent

        # Set the stylesheets
        self.set_styles({
            'open_comp_vals_pbutton': pbutton_composer,
            'open_outc_vals_pbutton': pbutton_composer,
            'open_feat_vals_pbutton': pbutton_composer,
            'open_metrics_graphs_pbutton': pbutton_composer,
            'open_metrics_tables_pbutton': pbutton_composer,
            'open_perm_pbutton': pbutton_composer,
            'open_dvh_pbutton': pbutton_composer,
            'open_ind_pbutton': pbutton_composer,
            'open_image_pbutton': pbutton_composer,
            'close_visualizer_pbutton': pbutton_composer})

        # Connect the fields with the event signals
        self.connect_signals()

        # Disable irrelevant tabs
        self.disable_tabs()

    def set_styles(
            self,
            key_value_pairs):
        """
        Set the element stylesheets from key-value pairs.

        Parameters
        ----------
        key_value_pairs : dict
            Dictionary with the field names (keys) and style sheets (values).
        """

        # Loop over the dictionary items
        for key, value in key_value_pairs.items():

            # Get the attribute and set the stylesheet
            getattr(self, key).setStyleSheet(value)

    def connect_signals(self):
        """Connect the fields with the event signals."""

        # Loop over the field names with 'clicked' events
        for key, value in {
                'open_comp_vals_pbutton': self.open_iter_graph,
                'open_outc_vals_pbutton': self.open_ntcp_graph,
                'open_feat_vals_pbutton': self.open_iter_graph,
                'open_metrics_graphs_pbutton': self.open_metrics_graph,
                'open_metrics_tables_pbutton': self.open_metrics_table,
                'open_perm_pbutton': self.open_permutation_importance_boxplot,
                'open_dvh_pbutton': self.open_dvh_graph,
                'open_ind_pbutton': self.open_dosimetrics_table,
                'open_image_pbutton': self.open_iter_graph
                }.items():

            # Connect the 'clicked' event
            getattr(self, key).clicked.connect(value)

        # Connect the 'clicked' event with the close button
        self.close_visualizer_pbutton.clicked.connect(self.close)

    def disable_tabs(self):
        """Disable irrelevant tabs."""

        # Get the datahub
        hub = Datahub()

        # Check if segmentation data is available
        if hub.segmentation is not None:

            # Get all conventional components
            cv_components = (
                get_conventional_constraints(hub.segmentation)
                + get_conventional_objectives(hub.segmentation))

            # Get the machine learning model-based components
            ml_components = (
                get_machine_learning_constraints(hub.segmentation)
                + get_machine_learning_objectives(hub.segmentation))

            # Get the radiobiological components
            rb_components = (
                get_radiobiological_constraints(hub.segmentation)
                + get_radiobiological_objectives(hub.segmentation))

        else:

            # Set the component tuples to the default value
            cv_components, ml_components, rb_components = (), (), ()

        # Check if the iteration plot buttons should be disabled
        if ((hub.state < 3 or
            (hub.optimization is not None and
             ('problem' not in hub.optimization or
              not hasattr(hub.optimization['problem'], 'tracker')
              or all(value == [] for value
                     in hub.optimization['problem'].tracker.values()))))):
            self.open_comp_vals_pbutton.setEnabled(False)
            self.open_outc_vals_pbutton.setEnabled(False)

        # Check if the iteration values button should be disabled
        if (not any(objective.display for objective in (
                *cv_components, *rb_components, *ml_components))):
            self.open_comp_vals_pbutton.setEnabled(False)

        # Check if the (N)TCP values button should be disabled
        if (not any(objective.display for objective in (
                rb_components + ml_components))):
            self.open_outc_vals_pbutton.setEnabled(False)

        # Check if the feature iterations button should be disabled
        if ((hub.state < 3 or
            (hub.optimization is not None and
             ('problem' not in hub.optimization or
              (not hasattr(hub.optimization['problem'], 'tracker')
               or all(value == [] for value
                      in hub.optimization['problem'].tracker.values()))))
             or all(objective.model_parameters.write_features is False
                    for objective in ml_components))):
            self.open_feat_vals_pbutton.setEnabled(False)

        # Check if the metrics tables and graphs buttons should be disabled
        if ((hub.state < 2 or
             (not hub.model_evaluations
              or len(hub.model_evaluations) == 0))):
            self.open_metrics_graphs_pbutton.setEnabled(False)
            self.open_metrics_tables_pbutton.setEnabled(False)

        # Check if the permutation importance button should be disabled
        if ((hub.state < 2 or
             (not hub.model_inspections
              or len(hub.model_inspections) == 0))):
            self.open_perm_pbutton.setEnabled(False)

        # Check if the plan evaluation buttons should be disabled
        if ((hub.state < 4 or
             (not hub.dose_histogram and not hub.dosimetrics))):
            self.open_dvh_pbutton.setEnabled(False)
            self.open_ind_pbutton.setEnabled(False)

        # Check if the CT/dose slice button should be disabled
        if ((hub.state < 1 or
             not hub.computed_tomography or not hub.segmentation)):
            self.open_image_pbutton.setEnabled(False)

    def open_iter_graph(self):
        """Open the iterative component value graph."""

        # Initialize the iterative component value graph
        plotter = IterGraph()

        # Open the view
        plotter.view()

    def open_ntcp_graph(self):
        """Open the (N)TCP graph."""

        # Initialize the (N)TCP graph
        plotter = NTCPGraph()

        # Open the view
        plotter.view()

    def open_metrics_graph(self):
        """Open the metrics graph."""

        # Initialize the metrics graph
        plotter = MetricsGraph()

        # Open the view
        plotter.view()

    def open_metrics_table(self):
        """Open the metrics table."""

        # Initialize the metrics table
        plotter = MetricsTable()

        # Open the view
        plotter.view()

    def open_permutation_importance_boxplot(self):
        """Open the permutation importance boxplot."""

        # Initialize the permutation importance boxplot
        plotter = PermutationImportanceBoxplot()

        # Open the view
        plotter.view()

    def open_dvh_graph(self):
        """Open the DVH graph."""

        # Initialize the DVH graph
        plotter = DVHGraph()

        # Open the view
        plotter.view()

    def open_dosimetrics_table(self):
        """Open the dosimetrics table."""

        # Initialize the dosimetrics table
        plotter = DosimetricsTable()

        # Open the view
        plotter.view()

    def launch(self):
        """Launch the visualizer."""

        if self.parent is None:

            # Set the window position
            self.position()

            # Show the visualization window
            self.show()

            # Run the application
            self.application.exec_()

        else:

            # Set the window position
            self.parent.visualization_window.position()

            # Show the window
            self.parent.visualization_window.show()

    def position(self):
        """Set the window position."""

        # Get the frame geometry
        geometry = self.frameGeometry()

        # Check if no parent has been passed
        if self.parent is None:

            # Get the screen number from the cursor position
            screen = QApplication.desktop().screenNumber(
                QApplication.desktop().cursor().pos())

            # Move the geometry center according to the application window
            geometry.moveCenter(
                QApplication.desktop().screenGeometry(screen).center())

            # Move the window to the top left of the geometry
            self.move(geometry.topLeft())

        else:

            # Move the geometry center according to the parent window
            geometry.moveCenter(self.parent.geometry().center())

            # Set the window geometry
            self.setGeometry(geometry)

    def close(self):
        """Close the visualizer."""

        # Check if no parent has been passed
        if self.parent is None:

            # Close the application
            self.application.quit()

            # Hide the window
            self.hide()

        else:

            # Hide the window
            self.hide()
