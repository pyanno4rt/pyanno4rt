"""Visualizer."""

# Author: Tim Ortkamp

# %% External package import

from math import isnan
from pandas import DataFrame
from pyqtgraph import mkQApp
from PyQt5.QtCore import QModelIndex
from PyQt5.QtWidgets import (
    QAbstractItemView, QApplication, QHeaderView, QMainWindow,
    QTableWidgetItem)

# %% Internal package import

from pyanno4rt.datahub import Datahub
from pyanno4rt.tools import (
    get_conventional_constraints, get_conventional_objectives,
    get_machine_learning_constraints, get_machine_learning_objectives,
    get_radiobiological_constraints, get_radiobiological_objectives)
from pyanno4rt.visualization.assets import resources_rc
from pyanno4rt.visualization._custom_styles import pbutton_composer
from pyanno4rt.visualization.custom_widgets import DVHWidget, SliceWidget
from pyanno4rt.visualization.design.visualizer import Ui_vis_window
from pyanno4rt.visualization.static import (
    DosimetricsTable, DVHGraph, IterGraph, MetricsGraph, MetricsTable,
    NTCPGraph, PermutationImportanceBoxplot)

# %% Class definition


class Visualizer(QMainWindow, Ui_vis_window):
    """
    Visualizer class.

    This class provides a visual analysis tool as standalone or for the \
    graphical user interface, including different types of visualizations.
    """

    def __init__(
            self,
            treatment_plan,
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

        # Initialize the base plan
        self.plan = treatment_plan

        # Initialize the widgets
        self.slice_widget = SliceWidget(self)
        self.dvh_widget = DVHWidget(self)

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

        # Add the widgets to the layouts
        self.image_top_widget_layout.insertWidget(0, self.slice_widget)
        self.dvh_top_widget_layout.insertWidget(0, self.dvh_widget)

        # Set the indicator table as read-only
        self.ind_table_widget.setEditTriggers(QAbstractItemView.NoEditTriggers)

        # Connect the fields with the event signals
        self.connect_signals()

    def mousePressEvent(
            self,
            event):
        """
        Set the mouse press event (overwrites the default event).

        Parameters
        ----------
        event : object of class :class:`~PyQt5.QtCore.QEvent`
            The object representing the event.
        """

        # Reset the table selection when clicking outside
        self.ind_table_widget.setCurrentIndex(QModelIndex())
        self.ind_table_widget.clearSelection()

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

    def set_enabled(
            self,
            field_names):
        """
        Enable multiple fields by their names.

        Parameters
        ----------
        field_names : tuple
            Tuple with the field names.
        """

        # Loop over the passed field names
        for name in field_names:

            # Get the attribute and enable the field
            getattr(self, name).setEnabled(True)

    def set_disabled(
            self,
            field_names):
        """
        Disable multiple fields by their names.

        Parameters
        ----------
        field_names : tuple
            Tuple with the field names.
        """

        # Loop over the passed field names
        for name in field_names:

            # Get the attribute and disable the field
            getattr(self, name).setEnabled(False)

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
                'open_image_pbutton': self.open_iter_graph,
                'close_visualizer_pbutton': self.close
                }.items():

            # Connect the 'clicked' event
            getattr(self, key).clicked.connect(value)

        # Loop over the field names with 'currentTextChanged' events
        for key, value in {
                'plane_cbox': self.slice_widget.change_orientation
                }.items():

            # Connect the 'currentTextChanged' event
            getattr(self, key).currentTextChanged.connect(value)

        # Loop over the field names with 'valueChanged' events
        for key, value in {
                'opacity_sbox': self.slice_widget.change_dose_opacity,
                'slice_selection_sbar': self.slice_widget.change_image_slice
                }.items():

            # Connect the 'valueChanged' event
            getattr(self, key).valueChanged.connect(value)

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

    def adjust_slider_by_orientation(self):
        """Adjust the slider for slice selection by the orientation."""

        # Create a mapping between planes and axes
        mapping = {'axial': 2, 'coronal': 0, 'sagittal': 1}

        # Check if the slice widget already stores a CT cube
        if self.slice_widget.ct_cube is not None:

            # Get the depth of the slice widget's CT cube
            plane_depth = self.slice_widget.ct_cube.shape[
                mapping[self.plane_cbox.currentText()]]

        else:

            # Get the depth of the current plan's CT cube
            plane_depth = self.plans[
                self.plan_ledit.text()].datahub.computed_tomography[
                    'cube_dimensions'][mapping[self.plane_cbox.currentText()]]

        # Set the range of the slice selection scroll bar
        self.slice_selection_sbar.setRange(0, plane_depth-1)

        # Set the initial scroll bar value
        self.slice_selection_sbar.setValue(int((plane_depth-1)/2))

    def add_images(self):
        """."""

        # Reset the images
        self.slice_widget.reset_images()

        # Check if the plan has already been configured
        if (all(getattr(self.plan, unit) is not None for unit in (
               'patient_loader', 'plan_generator', 'dose_info_generator'))
                and self.plan.datahub.state >= 1):

            # Add the CT cube to the slice widget
            self.slice_widget.add_ct()

            # Adjust the slider
            self.adjust_slider_by_orientation()

            # Update the slice widget images
            self.slice_widget.update_images()

            # Check if the plan has already been optimized
            if (self.plan.fluence_optimizer is not None
                    and 'optimized_dose' in self.plan.datahub.optimization
                    and self.plan.datahub.state >= 3):

                # Add the dose cube to the slice widget
                self.slice_widget.add_dose()

                # Update the slice widget images
                self.slice_widget.update_images()

    def add_dvh(self):
        """."""

        # Reset the DVH
        self.dvh_widget.reset_dvh()

        # Check if the plan has already been evaluated
        if (all(getattr(self.plan, unit) is not None for unit in (
                'dose_histogram', 'dosimetrics'))
                and self.plan.datahub.state == 4):

            # Add style and input data to the DVH widget
            self.dvh_widget.add_style_and_data(
                self.plan.datahub.dose_histogram)

            # Update the DVH plot
            self.dvh_widget.update_dvh()

    def add_indicators(self):
        """."""

        # Clear the plan quality indicator table
        self.ind_table_widget.clear()

        # Check if the plan has already been evaluated
        if (all(getattr(self.plan, unit) is not None for unit in (
                'dose_histogram', 'dosimetrics'))
                and self.plan.datahub.state == 4):

            # Get the display segments and metrics
            display_segments = self.plan.datahub.dosimetrics[
                'display_segments']
            display_metrics = self.plan.datahub.dosimetrics['display_metrics']

            # Get the filtered dosimetrics dictionary
            dosimetrics = {
                segment: {
                    metric: self.plan.datahub.dosimetrics[segment][metric]
                    for metric in self.plan.datahub.dosimetrics[segment]
                    if any(display_metric in metric
                           for display_metric in display_metrics)}
                for segment in display_segments}

            # Convert the dosimetrics dictionary into a dataframe
            dataframe = DataFrame(dosimetrics).transpose().astype(float)

            # Set the number of rows and columns
            self.ind_table_widget.setRowCount(0)
            self.ind_table_widget.setColumnCount(len(dataframe.columns))

            # Add the horizontal header labels
            self.ind_table_widget.setHorizontalHeaderLabels(dataframe.columns)

            # Set the resize mode for the horizontal section
            self.ind_table_widget.horizontalHeader().setSectionResizeMode(
                QHeaderView.Stretch)

            # Loop over the display segments
            for row, segment in enumerate(dataframe.index):

                # Insert a row
                self.ind_table_widget.insertRow(row)

                # Initialize the vertical header
                header = QTableWidgetItem()

                # Set the header text to the column name
                header.setText(segment)

                # Set the vertical header in the indicator table
                self.ind_table_widget.setVerticalHeaderItem(row, header)

                # Loop over the table columns
                for column, _ in enumerate(dataframe.columns):

                    # Initialize the table widget item
                    item = QTableWidgetItem(
                        str(round(dataframe.iloc[row, column], 4))
                        if not isnan(dataframe.iloc[row, column])
                        else '-')

                    self.ind_table_widget.setItem(row, column, item)

    def launch(self):
        """Launch the visualizer."""

        # Set the window position
        self.position()

        # Disable irrelevant tabs
        self.disable_tabs()

        # Add the CT/dose images
        self.add_images()

        # Add the DVH curves
        self.add_dvh()

        # Add the plan quality indicators
        self.add_indicators()

        # Show the visualization window
        self.show()

        # Check if no parent has been passed
        if self.parent is None:

            # Run the application
            self.application.exec_()

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
