"""Visualizer."""

# Author: Tim Ortkamp

# %% External package import

from math import isnan
from numpy import divide
from pandas import DataFrame
from pyqtgraph import mkQApp
from PyQt5.QtCore import QModelIndex
from PyQt5.QtWidgets import (
    QAbstractItemView, QApplication, QHeaderView, QMainWindow,
    QTableWidgetItem)

# %% Internal package import

from pyanno4rt.tools import (
    get_all_constraints, get_all_objectives, get_conventional_constraints,
    get_conventional_objectives, get_machine_learning_constraints,
    get_machine_learning_objectives, get_radiobiological_constraints,
    get_radiobiological_objectives)
from pyanno4rt.visualization.assets import resources_rc
from pyanno4rt.visualization.custom_widgets import (
    ComponentGraphWidget, DVHGraphWidget, OutcomeGraphWidget,
    PermutationImportanceWidget, SliceWidget)
from pyanno4rt.visualization.design.visualizer import Ui_visualization_window
from pyanno4rt.visualization.static import (
    ComponentGraph, DosimetricsTable, DVHGraph, MetricsGraph, MetricsTable,
    OutcomeGraph, PermutationImportanceBoxplot)
from pyanno4rt.visualization._custom_styles import (
    cbox, pbutton, tab_bright, tab_dark)

# %% Class definition


class Visualizer(QMainWindow, Ui_visualization_window):
    """
    Visualizer class.

    This class provides a visual analysis tool as standalone or for the \
    graphical user interface, including different types of visualizations.

    Parameters
    ----------
    treatment_plan : object of class \
        :class:`~pyanno4rt.base._treatment_plan.TreatmentPlan`
        The object used to represent the treatment plan.

    parent : object of class \
        :class:`~pyanno4rt.gui.windows._main_window.MainWindow`, default=None
        The object representing the parent window for embedding.

    Attributes
    ----------
    
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

        # Initialize the base plan
        self.plan = treatment_plan

        # Get the parent window
        self.parent = parent

        # Initialize the widgets
        self.comp_widget = ComponentGraphWidget(self)
        self.outc_widget = OutcomeGraphWidget(self)
        self.perm_widget = PermutationImportanceWidget(self)
        self.slice_widget = SliceWidget(self)
        self.dvh_widget = DVHGraphWidget(self)

        # Set the stylesheets
        self.set_styles({
            'categories_widget': tab_dark,
            'tab_problem': tab_bright,
            'tab_model': tab_bright,
            'tab_plan': tab_bright,
            'comp_background_cbox': cbox,
            'comp_gridcolor_cbox': cbox,
            'outc_background_cbox': cbox,
            'outc_gridcolor_cbox': cbox,
            'open_comp_graph_pbutton': pbutton,
            'open_outc_graph_pbutton': pbutton,
            'open_feat_vals_pbutton': pbutton,
            'open_metrics_graphs_pbutton': pbutton,
            'open_metrics_tables_pbutton': pbutton,
            'model_name_cbox': cbox,
            'domain_cbox': cbox,
            'perm_background_cbox': cbox,
            'perm_gridcolor_cbox': cbox,
            'open_perm_graph_pbutton': pbutton,
            'open_image_pbutton': pbutton,
            'dvh_background_cbox': cbox,
            'dvh_gridcolor_cbox': cbox,
            'open_dvh_graph_pbutton': pbutton,
            'open_ind_pbutton': pbutton,
            'close_visualizer_pbutton': pbutton})

        # Add the widgets to the layouts
        self.comp_graph_plot_widget_layout.insertWidget(0, self.comp_widget)
        self.outc_graph_plot_widget_layout.insertWidget(0, self.outc_widget)
        self.perm_graph_plot_widget_layout.insertWidget(0, self.perm_widget)
        self.image_top_widget_layout.insertWidget(0, self.slice_widget)
        self.dvh_graph_plot_widget_layout.insertWidget(0, self.dvh_widget)

        # Set the indicator table as read-only
        self.ind_table_widget.setEditTriggers(QAbstractItemView.NoEditTriggers)

        # Connect the fields with the event signals
        self.connect_signals()

    def mousePressEvent(
            self,
            _):
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
                'open_comp_graph_pbutton': self.open_component_graph,
                'open_outc_graph_pbutton': self.open_outcome_graph,
                'open_feat_vals_pbutton': self.open_component_graph,
                'open_metrics_graphs_pbutton': self.open_metrics_graph,
                'open_metrics_tables_pbutton': self.open_metrics_table,
                'open_perm_graph_pbutton': self.open_importance_boxplots,
                'open_dvh_graph_pbutton': self.open_dvh_graph,
                'open_ind_pbutton': self.open_dosimetrics_table,
                'open_image_pbutton': self.open_component_graph,
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

    def disable_buttons(self):
        """Disable irrelevant buttons."""

        # Get the datahub
        (computed_tomography, segmentation, optimization, model_evaluations,
         model_inspections, dose_histogram, dosimetrics, state) = (
             getattr(self.plan.datahub, attribute) for attribute in (
                 'computed_tomography', 'segmentation', 'optimization',
                 'model_evaluations', 'model_inspections', 'dose_histogram',
                 'dosimetrics', 'state'))

        # Check if segmentation data is available
        if segmentation is not None:

            # Get all conventional components
            cv_components = (
                get_conventional_constraints(segmentation)
                + get_conventional_objectives(segmentation))

            # Get the machine learning model-based components
            ml_components = (
                get_machine_learning_constraints(segmentation)
                + get_machine_learning_objectives(segmentation))

            # Get the radiobiological components
            rb_components = (
                get_radiobiological_constraints(segmentation)
                + get_radiobiological_objectives(segmentation))

        else:

            # Set the component tuples to the default value
            cv_components, ml_components, rb_components = (), (), ()

        # Check if the iteration plot buttons should be disabled
        if (state < 3 or
            (optimization is not None and
             ('problem' not in optimization or
              not hasattr(optimization['problem'], 'tracker')
              or all(value == [] for value
                     in optimization['problem'].tracker.values())))):
            self.open_comp_graph_pbutton.setEnabled(False)
            self.open_outc_graph_pbutton.setEnabled(False)

        # Check if the iteration values button should be disabled
        if (not any(objective.display for objective in (
                *cv_components, *rb_components, *ml_components))):
            self.open_comp_graph_pbutton.setEnabled(False)

        # Check if the (N)TCP values button should be disabled
        if (not any(objective.display for objective in (
                rb_components + ml_components))):
            self.open_outc_graph_pbutton.setEnabled(False)

        # Check if the feature iterations button should be disabled
        if (state < 3 or
            (optimization is not None and
             ('problem' not in optimization or
              (not hasattr(optimization['problem'], 'tracker')
               or all(value == [] for value
                      in optimization['problem'].tracker.values()))))
            or all(objective.model_parameters.write_features is False
                   for objective in ml_components)):
            self.open_feat_vals_pbutton.setEnabled(False)

        # Check if the metrics tables and graphs buttons should be disabled
        if (state < 2 or
            (model_evaluations is None
             or len(model_evaluations) == 0)):
            self.open_metrics_graphs_pbutton.setEnabled(False)
            self.open_metrics_tables_pbutton.setEnabled(False)

        # Check if the permutation importance button should be disabled
        if (state < 2 or
            (model_inspections is None
             or len(model_inspections) == 0)):
            self.open_perm_graph_pbutton.setEnabled(False)

        # Check if the plan evaluation buttons should be disabled
        if (state < 4 or
                (dose_histogram is None and dosimetrics is None)):
            self.open_dvh_graph_pbutton.setEnabled(False)
            self.open_ind_pbutton.setEnabled(False)

        # Check if the CT/dose slice button should be disabled
        if (state < 1 or
                computed_tomography is None or segmentation is None):
            self.open_image_pbutton.setEnabled(False)

    def add_component_tracks(self):
        """Add the component tracks to the widget."""

        # Reset the component graph
        self.comp_widget.reset_graph()

        # Get the segmentation and optimization data
        segmentation, optimization = (
            getattr(self.plan.datahub, attribute) for attribute in (
                'segmentation', 'optimization'))

        # Get all optimization components
        components = (
            get_all_objectives(segmentation)
            + get_all_constraints(segmentation))

        # Check if the plan has already been optimized
        if (self.plan.fluence_optimizer is not None
                and 'optimized_dose' in self.plan.datahub.optimization
                and self.plan.datahub.state >= 3):

            # Add style and data
            self.comp_widget.add_style_and_data(
                {component.track_id: (
                    optimization['problem'].tracker[component.track_id])
                    for component in components if component.display})

            # Update the component graph
            self.comp_widget.update_graph()

    def add_outcome_tracks(self):
        """Add the outcome tracks to the widget."""

        # Reset the outcome graph
        self.outc_widget.reset_graph()

        # Get the segmentation data
        segmentation = self.plan.datahub.segmentation

        # Get the outcome model-based components
        components = (
            get_machine_learning_constraints(segmentation)
            + get_machine_learning_objectives(segmentation)
            + get_radiobiological_constraints(segmentation)
            + get_radiobiological_objectives(segmentation))

        # Check if the plan has already been optimized
        if (self.plan.fluence_optimizer is not None
                and 'optimized_dose' in self.plan.datahub.optimization
                and self.plan.datahub.state >= 3) and len(components) > 0:

            # Get the tracker
            tracker = self.plan.datahub.optimization['problem'].tracker

            # Add style and data
            self.outc_widget.add_style_and_data(
                {component.track_id: component.translate(list(
                    divide(tracker[component.track_id], component.weight)))
                 for component in components})

            # Update the outcome graph
            self.outc_widget.update_graph()

    def add_importance_boxplots(self):
        """."""

        # Get the model inspections from the datahub
        model_inspections = self.plan.datahub.model_inspections

        # Reset the importance boxplots
        self.perm_widget.reset_boxplots()

        # Check if the plan has already been evaluated
        if (model_inspections is not None
                and len(model_inspections) > 0
                and self.plan.datahub.state >= 2):

            # Get the permutation importance data
            importances = {
                key: value['permutation_importance']
                for key, value in self.plan.datahub.model_inspections.items()
                if 'permutation_importance' in value}

            # Add the model names
            self.model_name_cbox.addItems(importances)

            # Get the number of features
            number_of_features = importances[
                self.model_name_cbox.currentText()]['Training'].shape[1]

            # Set the initial range for the top-k features
            self.num_features_sbox.setRange(1, number_of_features)

            # Set the initial value for the top-k features
            self.num_features_sbox.setValue(min(5, number_of_features))

            # Add style and data
            self.perm_widget.add_style_and_data(importances)

            # Update the importance boxplots
            self.perm_widget.update_boxplots()

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

    def add_dvh_values(self):
        """Add the DVH values to the widget."""

        # Reset the DVH graph
        self.dvh_widget.reset_graph()

        # Check if the plan has already been evaluated
        if (getattr(self.plan, 'dose_histogram') is not None
                and self.plan.datahub.state == 4):

            # Add style and data
            self.dvh_widget.add_style_and_data(
                self.plan.datahub.dose_histogram)

            # Update the DVH graph
            self.dvh_widget.update_graph()

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

    def open_component_graph(self):
        """Open the iterative component value graph."""

        # Set up the layout parameters
        inputs = {
            key: value for key, value in (
                ('title', self.comp_title_ledit.text()),
                ('titlesize', self.comp_titlesize_sbox.value()),
                ('xlabel', self.comp_x_ledit.text()),
                ('ylabel', self.comp_y_ledit.text()),
                ('labelsize', self.comp_labelsize_sbox.value()),
                ('linewidth', self.comp_linewidth_sbox.value()),
                ('ticksize', self.comp_ticksize_sbox.value()),
                ('legendsize', self.comp_legendsize_sbox.value()),
                ('background', self.comp_background_cbox.currentText()),
                ('gridlines', self.comp_gridlines_check.isChecked()),
                ('gridcolor', self.comp_gridcolor_cbox.currentText()))
            if value != ''}

        # Initialize the iterative component value graph
        plotter = ComponentGraph(**inputs)

        # Open the view
        plotter.view(
            self.plan,
            [item.name() for item in
             self.comp_widget.plot_widget.getPlotItem().curves
             if item.isVisible()])

    def open_outcome_graph(self):
        """Open the iterative outcome value graph."""

        # Set up the layout parameters
        inputs = {
            key: value for key, value in (
                ('title', self.outc_title_ledit.text()),
                ('titlesize', self.outc_titlesize_sbox.value()),
                ('xlabel', self.outc_x_ledit.text()),
                ('ylabel', self.outc_y_ledit.text()),
                ('labelsize', self.outc_labelsize_sbox.value()),
                ('linewidth', self.outc_linewidth_sbox.value()),
                ('ticksize', self.outc_ticksize_sbox.value()),
                ('legendsize', self.outc_legendsize_sbox.value()),
                ('background', self.outc_background_cbox.currentText()),
                ('gridlines', self.outc_gridlines_check.isChecked()),
                ('gridcolor', self.outc_gridcolor_cbox.currentText()))
            if value != ''}

        # Initialize the iterative outcome value graph
        plotter = OutcomeGraph(**inputs)

        # Open the view
        plotter.view(
            self.plan,
            [item.name() for item in
             self.outc_widget.plot_widget.getPlotItem().curves
             if item.isVisible()])

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

    def open_importance_boxplots(self):
        """Open the permutation importance boxplot."""

        # Set up the layout parameters
        inputs = {
            key: value for key, value in (
                ('title', self.perm_title_ledit.text()),
                ('titlesize', self.perm_titlesize_sbox.value()),
                ('xlabel', self.perm_x_ledit.text()),
                ('labelsize', self.perm_labelsize_sbox.value()),
                ('ticksize', self.perm_ticksize_sbox.value()),
                ('tickangle', self.perm_tickangle_sbox.value()),
                ('background', self.perm_background_cbox.currentText()),
                ('gridlines', self.perm_gridlines_check.isChecked()),
                ('gridcolor', self.perm_gridcolor_cbox.currentText()))
            if value != ''}

        # Initialize the permutation importance boxplot
        plotter = PermutationImportanceBoxplot(**inputs)

        # Open the view
        plotter.view(
            self.plan, self.model_name_cbox.currentText(),
            self.domain_cbox.currentText(), self.num_features_sbox.value())

    def open_dvh_graph(self):
        """Open the DVH graph."""

        # Set up the layout parameters
        inputs = {
            key: value for key, value in (
                ('title', self.dvh_title_ledit.text()),
                ('titlesize', self.dvh_titlesize_sbox.value()),
                ('xlabel', self.dvh_x_ledit.text()),
                ('ylabel', self.dvh_y_ledit.text()),
                ('labelsize', self.dvh_labelsize_sbox.value()),
                ('linewidth', self.dvh_linewidth_sbox.value()),
                ('ticksize', self.dvh_ticksize_sbox.value()),
                ('legendsize', self.dvh_legendsize_sbox.value()),
                ('background', self.dvh_background_cbox.currentText()),
                ('gridlines', self.dvh_gridlines_check.isChecked()),
                ('gridcolor', self.dvh_gridcolor_cbox.currentText()))
            if value != ''}

        # Initialize the DVH graph
        plotter = DVHGraph(**inputs)

        # Open the view
        plotter.view(
            self.plan,
            [item.name() for item in
             self.dvh_widget.plot_widget.getPlotItem().curves
             if item.isVisible()])

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

    def launch(self):
        """Launch the visualizer."""

        # Log a message about the visualizer launch
        self.plan.logger.display_info("Launching visualizer ...")

        # Set the window position
        self.position()

        # Disable irrelevant buttons
        self.disable_buttons()

        # Add the component tracks
        self.add_component_tracks()

        # Add the outcome tracks
        self.add_outcome_tracks()

        # Add the permutation importance boxplots
        self.add_importance_boxplots()

        # Add the CT/dose images
        self.add_images()

        # Add the DVH values
        self.add_dvh_values()

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
