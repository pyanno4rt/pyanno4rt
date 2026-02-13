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

from pyanno4rt.tools import (
    get_machine_learning_components, get_radiobiological_components)
from pyanno4rt.visualization.assets import resources_rc
from pyanno4rt.visualization.custom_widgets import (
    ComponentGraphWidget, DVHGraphWidget, FeatureGraphWidget,
    OutcomeGraphWidget, PermutationImportanceWidget, SliceWidget)
from pyanno4rt.visualization.designs.visualizer import Ui_visualization_window
from pyanno4rt.visualization.static import (
    ComponentGraph, DosimetricsTable, DVHGraph, FeatureGraph, MetricsGraph,
    MetricsTable, OutcomeGraph, PermutationImportanceBoxplot)
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

        # Initialize the application
        self.application = mkQApp("pyanno4rt")

        # Set the application style
        self.application.setStyle('Fusion')

        # Run the constructor from the superclass
        super().__init__()

        # Build the UI main window
        self.setupUi(self)

        # Initialize the base plan
        self.plan = treatment_plan

        # Get the parent window
        self.parent = parent

        # Initialize the widgets
        self.comp_widget = ComponentGraphWidget(self)
        self.outc_widget = OutcomeGraphWidget(self)
        self.feat_widget = FeatureGraphWidget(self)
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
            'open_comp_graph_pbutton': pbutton,
            'outc_background_cbox': cbox,
            'outc_gridcolor_cbox': cbox,
            'open_outc_graph_pbutton': pbutton,
            'feat_background_cbox': cbox,
            'feat_gridcolor_cbox': cbox,
            'open_feat_graph_pbutton': pbutton,
            'model_name_cbox': cbox,
            'model_cbox': cbox,
            'feature_cbox': cbox,
            'open_metrics_graphs_pbutton': pbutton,
            'open_metrics_tables_pbutton': pbutton,
            'perm_background_cbox': cbox,
            'perm_gridcolor_cbox': cbox,
            'open_perm_graph_pbutton': pbutton,
            'domain_cbox': cbox,
            'open_image_pbutton': pbutton,
            'dvh_background_cbox': cbox,
            'dvh_gridcolor_cbox': cbox,
            'open_dvh_graph_pbutton': pbutton,
            'open_ind_pbutton': pbutton,
            'close_visualizer_pbutton': pbutton})

        # Add the widgets to the layouts
        self.comp_graph_plot_widget_layout.insertWidget(0, self.comp_widget)
        self.feat_graph_plot_widget_layout.insertWidget(0, self.feat_widget)
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
                'open_feat_graph_pbutton': self.open_feature_graph,
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

        #
        computed_tomography = self.plan.patient_handler.computed_tomography
        segmentation = self.plan.patient_handler.segmentation
        problem = self.plan.fluence_optimizer.problem
        histogram = self.plan.dvh.histogram
        quantities = self.plan.dosimetrics.quantities

        # Get the datahub
        fluence_optimizer = self.plan.fluence_optimizer
        data_model_handler = self.plan.data_model_handler

        # Get the problem components
        components = problem.constraints + problem.objectives

        # Get the machine learning model-based components
        ml_components = get_machine_learning_components(components)

        # Get the radiobiological components
        rb_components = get_radiobiological_components(components)

        # Check if the iteration plot buttons should be disabled
        if (self.plan.state < 3 or
            (fluence_optimizer is not None and
             (not hasattr(fluence_optimizer, 'problem') or
              not hasattr(fluence_optimizer.problem, 'tracker')
              or all(value == [] for value
                     in fluence_optimizer.problem.tracker.values())))):
            self.open_comp_graph_pbutton.setEnabled(False)
            self.open_outc_graph_pbutton.setEnabled(False)

        # Check if the (N)TCP values button should be disabled
        if len(rb_components + ml_components) == 0:
            self.open_outc_graph_pbutton.setEnabled(False)

        # Check if the feature iterations button should be disabled
        if (self.plan.state < 3 or
            len(ml_components) == 0 or
            any(component.model.feature_calculator.feature_history is None
                for component in ml_components)):
            self.open_feat_graph_pbutton.setEnabled(False)

        # Check if the metrics tables and graphs buttons should be disabled
        if (self.plan.state < 2
            or data_model_handler is None
            or len(data_model_handler.models) == 0):
            self.open_metrics_graphs_pbutton.setEnabled(False)
            self.open_metrics_tables_pbutton.setEnabled(False)
            self.open_perm_graph_pbutton.setEnabled(False)

        # Check if the plan evaluation buttons should be disabled
        if (self.plan.state < 4 or
                (histogram is None and quantities is None)):
            self.open_dvh_graph_pbutton.setEnabled(False)
            self.open_ind_pbutton.setEnabled(False)

        # Check if the CT/dose slice button should be disabled
        if (self.plan.state < 1 or
                computed_tomography is None or segmentation is None):
            self.open_image_pbutton.setEnabled(False)

    def add_component_tracks(self):
        """Add the component tracks to the widget."""

        # Reset the component graph
        self.comp_widget.reset_graph()

        # Get the segmentation and optimization data
        problem = self.plan.fluence_optimizer.problem
        optimized_dose = self.plan.fluence_optimizer.optimized_dose

        # Get the problem components
        components = problem.constraints + problem.objectives

        # Check if the plan has already been optimized
        if (self.plan.fluence_optimizer is not None
                and optimized_dose is not None
                and all(value != [] for value in problem.tracker.values())
                and self.plan.state >= 3):

            # Add style and data
            self.comp_widget.add_style_and_data(
                {component.track_id: (
                    problem.tracker[component.track_id])
                    for component in components})

            # Update the component graph
            self.comp_widget.update_graph()

    def add_outcome_tracks(self):
        """Add the outcome tracks to the widget."""

        # Reset the outcome graph
        self.outc_widget.reset_graph()

        # Get the segmentation and optimization data
        problem = self.plan.fluence_optimizer.problem
        optimized_dose = self.plan.fluence_optimizer.optimized_dose

        # Get the problem components
        components = problem.constraints + problem.objectives

        # Get the outcome model-based components
        components = (
            get_machine_learning_components(components)
            + get_radiobiological_components(components))

        # Check if the plan has already been optimized
        if (self.plan.fluence_optimizer is not None
                and optimized_dose is not None
                and self.plan.state >= 3
                and len(components) > 0):

            # Add style and data
            self.outc_widget.add_style_and_data(
                {component.track_id: component.translate(
                    problem.tracker[component.track_id])
                 for component in components})

            # Update the outcome graph
            self.outc_widget.update_graph()

    def add_feature_tracks(self):
        """Add the feature tracks to the widget."""

        # Reset the feature graph
        self.feat_widget.reset_graph()

        # Get the segmentation and optimization data
        problem = self.plan.fluence_optimizer.problem

        # Get the ML model-based components
        components = get_machine_learning_components(
            problem.constraints + problem.objectives)

        # Check if the plan has already been optimized
        if (self.plan.state >= 3 or
            any(component.model.feature_calculator.feature_history is None
                for component in components)):

            # Get the feature histories
            histories = {label: values for label, values in {
                component.model.label: getattr(
                    component.model.feature_calculator,
                    'feature_history') for component in components}.items()
                if values}

            # Check if any history has been recorded
            if len(histories) > 0:

                # Get the outcome data
                outcomes = {
                    component.model.label: component.translate(
                        problem.tracker[component.track_id])
                    for component in components}

                # Add the model names
                self.model_cbox.addItems(list(histories.keys()))

                # Add the feature names
                self.feature_cbox.addItems(
                    list(histories[self.model_cbox.currentText()]))

                # Add style and data
                self.feat_widget.add_style_and_data(histories, outcomes)

                # Update the feature graph
                self.feat_widget.update_graph()

    def add_importance_boxplots(self):
        """."""

        # Reset the importance boxplots
        self.perm_widget.reset_boxplots()

        # Check if the plan has already been inspected
        if (self.plan.state >= 2 and
                self.plan.data_model_handler is not None and
                len(self.plan.data_model_handler.models) > 0):

            #
            importances = {
                model.label: model.inspector.results['permutation_importances']
                for model in self.plan.data_model_handler.models
                if model.inspector is not None}

            #
            if len(importances) > 0:

                # Add the model names
                self.model_name_cbox.addItems(importances)

                # Get the number of features
                number_of_features = importances[
                    self.model_name_cbox.currentText()]['Full'].shape[1]

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
               'patient_handler', 'plan_handler', 'dose_handler'))
                and self.plan.state >= 1):

            # Add the CT cube to the slice widget
            self.slice_widget.add_ct()

            # Adjust the slider
            self.adjust_slider_by_orientation()

            # Update the slice widget images
            self.slice_widget.update_images()

            # Check if the plan has already been optimized
            if (self.plan.fluence_optimizer is not None
                    and self.plan.fluence_optimizer.optimized_dose is not None
                    and self.plan.state >= 3):

                # Add the dose cube to the slice widget
                self.slice_widget.add_dose()

                # Update the slice widget images
                self.slice_widget.update_images()

    def add_dvh_values(self):
        """Add the DVH values to the widget."""

        # Reset the DVH graph
        self.dvh_widget.reset_graph()

        # Check if the plan has already been evaluated
        if getattr(self.plan, 'dvh') is not None and self.plan.state == 4:

            # Add style and data
            self.dvh_widget.add_style_and_data(self.plan.dvh.histogram)

            # Update the DVH graph
            self.dvh_widget.update_graph()

    def add_indicators(self):
        """."""

        # Clear the plan quality indicator table
        self.ind_table_widget.clear()

        # Check if the plan has already been evaluated
        if (all(getattr(self.plan, attribute) is not None for attribute in (
                'dvh', 'dosimetrics')) and self.plan.state == 4):

            # Convert the quantities dictionary into a dataframe
            dataframe = DataFrame(
                self.plan.dosimetrics.quantities).transpose().astype(float)

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

    def open_feature_graph(self):
        """Open the iterative feature value graph."""

        # Set up the layout parameters
        inputs = {
            key: value for key, value in (
                ('title', self.feat_title_ledit.text()),
                ('titlesize', self.feat_titlesize_sbox.value()),
                ('xlabel', self.feat_x_ledit.text()),
                ('ylabel', self.feat_y_ledit.text()),
                ('labelsize', self.feat_labelsize_sbox.value()),
                ('linewidth', self.feat_linewidth_sbox.value()),
                ('ticksize', self.feat_ticksize_sbox.value()),
                ('legendsize', self.feat_legendsize_sbox.value()),
                ('background', self.feat_background_cbox.currentText()),
                ('gridlines', self.feat_gridlines_check.isChecked()),
                ('gridcolor', self.feat_gridcolor_cbox.currentText()))
            if value != ''}

        # Initialize the iterative feature value graph
        plotter = FeatureGraph(**inputs)

        # Open the view
        plotter.view(
            (self.feature_cbox.currentText(),
             self.feat_widget.histories[self.model_cbox.currentText()][
                self.feature_cbox.currentText()]),
            (self.model_cbox.currentText(),
             self.feat_widget.outcomes[self.model_cbox.currentText()]
             if self.show_ntcp_check.isChecked() else None))

    def open_metrics_graph(self):
        """Open the metrics graph."""

        # Initialize the metrics graph
        plotter = MetricsGraph()

        # Open the view
        plotter.view(self.plan, self.model_name_cbox.currentText())

    def open_metrics_table(self):
        """Open the metrics table."""

        # Initialize the metrics table
        plotter = MetricsTable()

        # Open the view
        plotter.view(self.plan, self.model_name_cbox.currentText())

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
        plotter.view(self.plan)

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
            plane_depth = self.plan.patient_handler.computed_tomography[
                    'cube_dimensions'][mapping[self.plane_cbox.currentText()]]

        # Set the range of the slice selection scroll bar
        self.slice_selection_sbar.setRange(0, plane_depth-1)

        # Set the initial scroll bar value
        self.slice_selection_sbar.setValue(int((plane_depth-1)/2))

    def launch(self):
        """Launch the visualizer."""

        # Set the window position
        self.position()

        # Disable irrelevant buttons
        self.disable_buttons()

        # Add the component tracks
        self.add_component_tracks()

        # Add the outcome tracks
        self.add_outcome_tracks()

        # Add the feature tracks
        self.add_feature_tracks()

        # Add the permutation importance boxplots
        self.add_importance_boxplots()

        # Add the CT/dose images
        self.add_images()

        # Add the DVH values
        self.add_dvh_values()

        # Add the plan quality indicators
        self.add_indicators()

        # Check if no parent has been passed
        if self.parent is None:

            # Show the maximized visualization window
            self.showMaximized()

            # Run the application
            self.application.exec_()

        else:

            # Show the visualization window
            self.show()

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

            # Set the window size
            self.resize(self.parent.size())

    def close(self):
        """Close the visualizer."""

        # Check if no parent has been passed
        if self.parent is None:

            # Close the application
            self.application.quit()

        # Hide the window
        self.hide()
