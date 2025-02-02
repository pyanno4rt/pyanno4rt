"""Main window."""

# Author: Tim Ortkamp

# %% External package import

from webbrowser import open as webopen

from functools import partial, reduce
from importlib.metadata import version
from json import dumps, loads
from logging import Handler
from os.path import abspath, dirname
from numpy import zeros
from PyQt5.QtCore import pyqtSignal, QEvent, QObject, Qt, QThread
from PyQt5.QtGui import QCursor, QIcon, QMovie, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QFrame, QHeaderView, QLabel,
    QListWidgetItem, QMainWindow, QMenu, QMessageBox, QPushButton, QSpinBox)

# %% Internal package import

from pyanno4rt.base import TreatmentPlan
from pyanno4rt.gui.assets import resources_rc
from pyanno4rt.gui.compilations.main_window import Ui_main_window
from pyanno4rt.gui.custom_widgets import (
    CheckableComboBox, DVHWidget, SliceWidget)
from pyanno4rt.gui.styles._custom_styles import (
    cbox, ledit, pbutton_menu, pbutton_composer, pbutton_statusbar,
    pbutton_workflow, sbox, selector, tab, tbutton_composer, tbutton_workflow)
from pyanno4rt.gui.windows import (
    CompareWindow, InfoWindow, LogWindow, PlanCreationWindow, SettingsWindow,
    SplashScreenWindow, TreeWindow)
from pyanno4rt.gui.windows.components import component_window_map
from pyanno4rt.optimization.components import (
    ConventionalComponent, component_map, MachineLearningComponent,
    RadiobiologicalComponent)
from pyanno4rt.optimization.methods import method_map
from pyanno4rt.optimization.solvers import solver_map
from pyanno4rt.tools import (
    add_square_brackets, apply, copycat, get_machine_learning_constraints,
    get_machine_learning_objectives, load_list_from_file,
    load_segments_from_path, snapshot, string_to_numeric)

# %% Class definition


class MainWindow(QMainWindow, Ui_main_window):
    """
    Main window for the GUI.

    This class sets up the main window for the graphical user interface, \
    including the main surface with all input/control elements.

    Parameters
    ----------
    treatment_plan : object of class \
        :class:`~pyanno4rt.base._treatment_plan.TreatmentPlan`, default=None
        The object used to represent the initial treatment plan.

    application : object of class :class:`~PyQt5.QtWidgets.QApplication`, \
        default=None
        The object used to represent the widget-based Qt application.
    """

    def __init__(
            self,
            treatment_plan=None,
            application=None):

        # Run the constructor from the superclass
        super().__init__()

        # Build the UI main window
        self.setupUi(self)

        # Initialize the splash screen window
        self.splash_screen_window = SplashScreenWindow()

        # Set the position of the splash screen and main window
        self.splash_screen_window.position()
        self.position()

        # Show the splash screen window
        self.splash_screen_window.show()

        # Get the application
        self.application = application

        # Initialize the plan dictionary
        self.plans = {}

        # Initialize the last selected plan
        self.last_selection = ''

        # Initialize the current component window
        self.current_component_window = None

        # Initialize the plan component dictionary
        self.plan_components = {}

        # Initialize the segment dictionary
        self.segments = {}

        # Initialize the data presets
        self.data_presets = {}

        # Initialize the thread and the worker
        self.thread = None
        self.worker = None

        # Create a event process loop to run the splash screen
        for i in range(10000):
            self.application.processEvents()

        # Start the splash screen progress bar
        self.splash_screen_window.progress()

        # Initialize the child windows
        self.plan_creation_window = PlanCreationWindow(self)
        self.settings_window = SettingsWindow(self)
        self.info_window = InfoWindow(self)
        self.compare_window = CompareWindow(self)
        self.config_window = TreeWindow('Plan Configuration Viewer', self)
        self.datahub_window = TreeWindow('Datahub Content Viewer', self)
        self.log_window = LogWindow(self)

        # Initialize the custom widgets
        self.slice_widget = SliceWidget(self)
        self.dvh_widget = DVHWidget(self)

        # Insert the custom widgets into the viewer layouts
        self.tab_slices_layout.insertWidget(0, self.slice_widget)
        self.tab_dvh_layout.insertWidget(0, self.dvh_widget)

        # Add the optimization methods to the method combo box
        self.method_cbox.addItems(list(method_map.keys()))
        self.method_cbox.model().sort(0)
        self.method_cbox.setCurrentText('weighted-sum')

        # Add the solvers to the solver combo box
        self.solver_cbox.addItems(list(solver_map.keys()))
        self.solver_cbox.model().sort(0)
        self.solver_cbox.setCurrentText('scipy')
        self.update_by_solver()

        # Initialize the custom combo box for the display segments
        self.display_segments_cbox = CheckableComboBox()
        self.horizontal_layout.addWidget(self.display_segments_cbox)

        # Initialize the custom combo box for the display metrics
        self.display_metrics_cbox = CheckableComboBox()
        self.horizontal_layout.addWidget(self.display_metrics_cbox)

        # Add the display metrics items
        self.display_metrics_cbox.addItems(
            ['mean', 'std', 'max', 'min', 'Dx', 'Vx', 'CI', 'HI'])

        # Get the base input dictionaries
        self.base_configuration = self.transform_configuration_to_dict()
        self.base_optimization = self.transform_optimization_to_dict()
        self.base_evaluation = self.transform_evaluation_to_dict()

        # Add the dropdown menu to the components 'plus' button
        self.add_dropdown_to_components()

        # Configure the status bar
        self.configure_status_bar()

        # Set the stylesheets
        self.set_styles({
            'composer_widget': tab,
            'tab_workflow': tab,
            'viewer_widget': tab,
            'load_pbutton': pbutton_menu,
            'save_pbutton': pbutton_menu,
            'drop_pbutton': pbutton_menu,
            'plan_select_cbox': selector,
            'settings_pbutton': pbutton_menu,
            'info_pbutton': pbutton_menu,
            'exit_pbutton': pbutton_menu,
            'plan_ledit': ledit,
            'log_level_cbox': cbox,
            'modality_cbox': cbox,
            'nfx_sbox': sbox,
            'img_path_ledit': ledit,
            'img_path_tbutton': tbutton_composer,
            'dose_path_ledit': ledit,
            'dose_path_tbutton': tbutton_composer,
            'dose_res_ledit_x': ledit,
            'dose_res_ledit_y': ledit,
            'dose_res_ledit_z': ledit,
            'update_configuration_pbutton': pbutton_composer,
            'reset_configuration_pbutton': pbutton_composer,
            'clear_configuration_pbutton': pbutton_composer,
            'components_plus_tbutton': tbutton_composer,
            'components_minus_tbutton': tbutton_composer,
            'components_edit_tbutton': tbutton_composer,
            'method_cbox': cbox,
            'solver_cbox': cbox,
            'algorithm_cbox': cbox,
            'init_strat_cbox': cbox,
            'init_fluence_ledit': ledit,
            'init_fluence_tbutton': tbutton_composer,
            'ref_plan_cbox': cbox,
            'lower_var_ledit': ledit,
            'lower_var_tbutton': tbutton_composer,
            'upper_var_ledit': ledit,
            'upper_var_tbutton': tbutton_composer,
            'max_iter_sbox': sbox,
            'tolerance_ledit': ledit,
            'update_optimization_pbutton': pbutton_composer,
            'reset_optimization_pbutton': pbutton_composer,
            'clear_optimization_pbutton': pbutton_composer,
            'dvh_type_cbox': cbox,
            'n_points_sbox': sbox,
            'ref_vol_ledit': ledit,
            'ref_dose_ledit': ledit,
            'display_segments_cbox': cbox,
            'display_metrics_cbox': cbox,
            'update_evaluation_pbutton': pbutton_composer,
            'reset_evaluation_pbutton': pbutton_composer,
            'clear_evaluation_pbutton': pbutton_composer,
            'configure_pbutton': pbutton_workflow,
            'model_pbutton': pbutton_workflow,
            'optimize_pbutton': pbutton_workflow,
            'evaluate_pbutton': pbutton_workflow,
            'visualize_pbutton': pbutton_workflow,
            'actions_show_config_tbutton': tbutton_workflow,
            'actions_show_datahub_tbutton': tbutton_workflow,
            'actions_show_log_tbutton': tbutton_workflow,
            'actions_export_to_pyfile_tbutton': tbutton_workflow,
            'baseline_ledit': ledit,
            'reference_cbox': cbox,
            'compare_pbutton': pbutton_workflow,
            'compare_show_config_tbutton': tbutton_workflow,
            'compare_show_datahub_tbutton': tbutton_workflow,
            'compare_show_log_tbutton': tbutton_workflow,
            'compare_export_to_pyfile_tbutton': tbutton_workflow,
            'plane_cbox': cbox,
            'opacity_sbox': sbox})

        # Loop over the QComboBox and QSpinBox elements
        for box in (
                'log_level_cbox', 'modality_cbox', 'nfx_sbox', 'method_cbox',
                'solver_cbox', 'algorithm_cbox', 'init_strat_cbox',
                'ref_plan_cbox', 'max_iter_sbox', 'dvh_type_cbox',
                'n_points_sbox', 'display_segments_cbox',
                'display_metrics_cbox', 'reference_cbox', 'plane_cbox',
                'opacity_sbox'):

            # Install the custom event filter
            getattr(self, box).installEventFilter(self)

        # Set the line edit cursor positions to zero
        self.set_zero_line_cursor((
            'img_path_ledit', 'dose_path_ledit', 'init_fluence_ledit',
            'lower_var_ledit', 'upper_var_ledit', 'ref_vol_ledit',
            'ref_dose_ledit', 'baseline_ledit'))

        # Loop over the tab widgets
        for widget in ('composer_widget', 'tab_workflow', 'viewer_widget'):

            # Set the initial tab index
            getattr(self, widget).setCurrentIndex(0)

        # Loop over the composer subwidgets
        for i in range(3):

            # Disable the subwidget
            self.composer_widget.widget(i).setEnabled(False)

        # Adjust the spacing
        self.components_lwidget.setSpacing(4)

        # Overwrite the wheel event
        self.components_lwidget.wheelEvent = lambda event: None

        # Disable some fields
        self.set_disabled((
            'save_pbutton', 'drop_pbutton', 'update_configuration_pbutton',
            'update_optimization_pbutton', 'update_evaluation_pbutton',
            'reset_configuration_pbutton', 'reset_optimization_pbutton',
            'reset_evaluation_pbutton', 'configure_pbutton', 'model_pbutton',
            'optimize_pbutton', 'evaluate_pbutton', 'visualize_pbutton',
            'actions_show_config_tbutton', 'actions_show_datahub_tbutton',
            'actions_show_log_tbutton', 'actions_export_to_pyfile_tbutton',
            'compare_show_config_tbutton', 'compare_show_datahub_tbutton',
            'compare_show_log_tbutton', 'compare_export_to_pyfile_tbutton',
            'compare_pbutton', 'reference_cbox', 'components_minus_tbutton',
            'components_edit_tbutton', 'init_fluence_ledit',
            'init_fluence_tbutton', 'ref_plan_cbox', 'plane_cbox',
            'opacity_sbox', 'slice_selection_sbar', 'stop_thread_pbutton'))

        # Connect the event signals
        self.connect_signals()

        # Check if an initial treatment plan has been specified
        if treatment_plan:

            # Set the initial treatment plan
            self.set_initial_plan(treatment_plan)

        # Show the window
        self.show()

        # Close the splash screen window
        self.splash_screen_window.close()

    def eventFilter(
            self,
            source,
            event):
        """
        Filter the events (overwrites the default event filter).

        Parameters
        ----------
        source : object of class :class:`~PyQt5.QtWidgets`
            The object representing the event source.

        event : object of class :class:`~PyQt5.QtCore.QEvent`
            The object representing the event.

        Returns
        -------
        bool or object of class :class:`~PyQt5.QtCore.QEvent`
            Boolean value or event object depending on the filter.
        """

        # Check if a mouse wheel event applies to QComboBox or QSpinBox
        if (event.type() == QEvent.Wheel and
                isinstance(source, (QComboBox, QSpinBox))):

            # Filter the event by returning True
            return True

        # Else, return the unfiltered event
        return super().eventFilter(source, event)

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

        # Check if no item of the components list widget has been clicked
        if not self.components_lwidget.indexAt(event.pos()).isValid():

            # Clear the item selection
            self.components_lwidget.clearSelection()

            # Disable the 'minus' and 'edit' buttons
            self.set_disabled((
                'components_minus_tbutton', 'components_edit_tbutton'))

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

            # Check if the key does not refer to a tab widget
            if key in ('composer_widget', 'tab_workflow', 'viewer_widget'):

                # Get the tab bar of the attribute and set the stylesheet
                getattr(self, key).tabBar().setStyleSheet(value)

            else:

                # Get the attribute and set the stylesheet
                getattr(self, key).setStyleSheet(value)

    def set_zero_line_cursor(
            self,
            field_names):
        """
        Set the line edit cursor positions to zero.

        Parameters
        ----------
        field_names : tuple
            Tuple with the field names.
        """

        # Loop over the passed field names
        for name in field_names:

            # Get the attribute and set the cursor position to zero
            getattr(self, name).setCursorPosition(0)

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
                'load_pbutton': self.load_tpi,
                'save_pbutton': self.save_tpi,
                'settings_pbutton': self.open_settings_window,
                'info_pbutton': self.open_info_window,
                'exit_pbutton': self.open_question_dialog,
                'drop_pbutton': self.open_question_dialog,
                'img_path_tbutton': self.add_imaging_path,
                'dose_path_tbutton': self.add_dose_matrix_path,
                'update_configuration_pbutton': self.open_question_dialog,
                'reset_configuration_pbutton': self.open_question_dialog,
                'clear_configuration_pbutton': self.open_question_dialog,
                'components_minus_tbutton': self.remove_component,
                'components_edit_tbutton': self.edit_component,
                'init_fluence_tbutton': self.add_initial_fluence_vector,
                'lower_var_tbutton': self.add_lower_var_bounds,
                'upper_var_tbutton': self.add_upper_var_bounds,
                'update_optimization_pbutton': self.open_question_dialog,
                'reset_optimization_pbutton': self.open_question_dialog,
                'clear_optimization_pbutton': self.open_question_dialog,
                'update_evaluation_pbutton': self.open_question_dialog,
                'reset_evaluation_pbutton': self.open_question_dialog,
                'clear_evaluation_pbutton': self.open_question_dialog,
                'configure_pbutton': self.start_configure,
                'model_pbutton': self.start_model,
                'optimize_pbutton': self.start_optimize,
                'evaluate_pbutton': self.start_evaluate,
                'visualize_pbutton': self.visualize,
                'actions_show_config_tbutton': self.open_configuration_window,
                'actions_show_datahub_tbutton': self.open_datahub_window,
                'actions_show_log_tbutton': self.open_log_window,
                'actions_export_to_pyfile_tbutton': self.export_to_pyfile,
                'compare_pbutton': self.open_compare_window,
                'compare_show_config_tbutton': self.open_configuration_window,
                'compare_show_datahub_tbutton': self.open_datahub_window,
                'compare_show_log_tbutton': self.open_log_window,
                'compare_export_to_pyfile_tbutton': self.export_to_pyfile,
                'stop_thread_pbutton': self.stop_thread_by_user,
                'github_pbutton': self.open_github_link,
                'rtd_pbutton': self.open_rtd_link,
                'pypi_pbutton': self.open_pypi_link
                }.items():

            # Connect the 'clicked' event
            getattr(self, key).clicked.connect(value)

        # Loop over the field names with 'currentIndexChanged' events
        for key, value in {
                'plan_select_cbox': self.update_reference_plans,
                'method_cbox': self.update_by_method,
                'solver_cbox': self.update_by_solver
                }.items():

            # Connect the 'currentIndexChanged' event
            getattr(self, key).currentIndexChanged.connect(value)

        # Loop over the field names with 'currentTextChanged' events
        for key, value in {
                'plan_select_cbox': self.select_plan,
                'init_strat_cbox': self.update_by_initial_strategy,
                'ref_plan_cbox': self.update_by_reference,
                'plane_cbox': self.slice_widget.change_orientation
                }.items():

            # Connect the 'currentTextChanged' event
            getattr(self, key).currentTextChanged.connect(value)

        # Loop over the field names with 'textChanged' events
        for key, value in {
                'init_fluence_ledit': self.update_by_initial_fluence
                }.items():

            # Connect the 'textChanged' event
            getattr(self, key).textChanged.connect(value)

        # Loop over the field names with 'valueChanged' events
        for key, value in {
                'opacity_sbox': self.slice_widget.change_dose_opacity,
                'slice_selection_sbar': self.slice_widget.change_image_slice
                }.items():

            # Connect the 'valueChanged' event
            getattr(self, key).valueChanged.connect(value)

        # Loop over the field names with 'itemClicked' events
        for key, value in {
                'components_lwidget': (lambda: self.set_enabled((
                    'components_minus_tbutton', 'components_edit_tbutton')))
                }.items():

            # Connect the 'itemClicked' event
            getattr(self, key).itemClicked.connect(value)

    def set_initial_plan(
            self,
            treatment_plan):
        """
        Set up the initial treatment plan(s).

        Parameters
        ----------
        treatment_plan : object or list of objects of class \
            :class:`~pyanno4rt.base._treatment_plan.TreatmentPlan`
            The object used to represent the initial treatment plan(s).
        """

        # Check if the input is a single treatment plan
        if type(treatment_plan).__name__ is TreatmentPlan.__name__:

            # Activate the treatment plan
            self.activate(treatment_plan)

        # Else, check if the input is a list of treatment plans
        elif (isinstance(treatment_plan, list) and
              all(type(plan).__name__ is TreatmentPlan.__name__
                  for plan in treatment_plan)):

            # Activate each treatment plan
            apply(self.activate, treatment_plan)

            # Set the selector index to the first element
            self.plan_select_cbox.setCurrentIndex(0)

    def activate(
            self,
            treatment_plan):
        """
        Activate a treatment plan instance.

        Parameters
        ----------
        treatment_plan : object of class \
            :class:`~pyanno4rt.base._treatment_plan.TreatmentPlan`
            The object used to represent the treatment plan.
        """

        # Get the treatment plan label
        label = treatment_plan.configuration['label']

        # Check if the label is not yet included in the selector
        if label not in (
                self.plan_select_cbox.itemText(i)
                for i in range(self.plan_select_cbox.count())):

            # Add label and treatment plan to the plan dictionary
            self.plans[label] = treatment_plan

            # Add the label to the plan component dictionary
            self.plan_components[label] = {}

            # Add the label to the selector
            self.plan_select_cbox.insertItem(0, label)

            # Temporarily remove the 'Create new plan' item
            self.plan_select_cbox.removeItem(self.plan_select_cbox.count()-1)

            # Sort the items in the selector alphabetically
            self.plan_select_cbox.model().sort(0)

            # Initialize the icon object
            icon = QIcon()

            # Add the pixmap to the icon
            icon.addPixmap(
                QPixmap(":/lightred_icons/icons_lightred/plus-square.svg"),
                QIcon.Normal, QIcon.Off)

            # Insert the 'Create new plan' item again
            self.plan_select_cbox.insertItem(
                self.plan_select_cbox.count(), icon, 'Create new plan')

            # Select the label
            self.plan_select_cbox.setCurrentText(label)

            # Initialize the console window log handler
            handler = ConsoleWindowLogHandler()

            # Connect the handler with the status bar
            handler.stream.connect(self.status_bar.showMessage)

            # Add the handler to the treatment plan logger
            treatment_plan.logger.logger.addHandler(handler)

        else:

            # Select the label
            self.plan_select_cbox.setCurrentText(label)

    def load_tpi(self):
        """Load a treatment plan."""

        # Get the folder path
        path = QFileDialog.getExistingDirectory(self, 'Select a directory')

        # Check if a path has been selected
        if path:

            try:

                # Activate the treatment plan
                self.activate(copycat(TreatmentPlan, path))

            except FileNotFoundError as exception:

                # Show a warning message box
                QMessageBox.warning(
                    self, "pyanno4rt",
                    "Exception occurred during treatment plan loading - "
                    "please check that the path leads to a snapshot! \n\n"
                    f"{type(exception).__name__}: {str(exception)}")

    def save_tpi(self):
        """Save a treatment plan."""

        # Get the folder path
        path = QFileDialog.getExistingDirectory(self, 'Select a directory')

        # Check if a path has been selected
        if path:

            # Get the additional parameters from the settings window
            includes = self.settings_window.current[3]

            # Make a snapshot of the treatment plan
            snapshot(self.plans[self.plan_ledit.text()],
                     ''.join((path, '/')), *includes)

    def drop_tpi(self):
        """Drop a treatment plan."""

        # Get the treatment plan label
        label = self.plan_ledit.text()

        # Check if the label is included in the plan dictionary
        if label in self.plans:

            # Delete label and treatment plan from the plan dictionary
            del self.plans[label]

            # Reset the selector
            self.plan_select_cbox.setCurrentIndex(-1)

            # Loop over the selector items
            for i in range(self.plan_select_cbox.count()):

                # Check if the item text equals the label
                if label == self.plan_select_cbox.itemText(i):

                    # Remove the item
                    self.plan_select_cbox.removeItem(i)

            # Check if the first item text is 'Create new plan'
            if self.plan_select_cbox.itemText(0) == 'Create new plan':

                # Insert the empty item
                self.plan_select_cbox.insertItem(0, '')

                # Set the selector to the empty item
                self.plan_select_cbox.setCurrentIndex(0)

            # Clear the composer tabs
            self.clear_configuration()
            self.clear_optimization()
            self.clear_evaluation()

            # Clear the reference combo box for comparison
            self.reference_cbox.clear()

    def select_plan(self):
        """Select a treatment plan."""

        # Get the selected item text
        selection = self.plan_select_cbox.currentText()

        # Set the baseline plan for comparison
        self.baseline_ledit.setText(selection)

        # Check if 'Create new plan' has been selected
        if selection == 'Create new plan':

            # Set the selector to the last selection
            self.plan_select_cbox.setCurrentText(self.last_selection)

            # Open the plan creation window
            self.open_plan_creation_window()

        # Check if a label item has been selected
        elif selection != '':

            # Loop over the selector items
            for i in range(self.plan_select_cbox.count()):

                # Check if the item text is empty
                if self.plan_select_cbox.itemText(i) == '':

                    # Remove the item
                    self.plan_select_cbox.removeItem(i)

            # Update the last selected item
            self.last_selection = selection

            # Update the treatment plan label
            self.plan_ledit.setText(selection)

            # Set the composer tabs
            self.set_configuration()
            self.set_optimization()
            self.set_evaluation()

            # Reset the slice widget
            self.slice_widget.reset_images()

            # Reset the DVH widget
            self.dvh_widget.reset_dvh()

            # Update the log output
            self.log_window.update_log_output()

            # Set the line edit cursor positions to zero
            self.set_zero_line_cursor((
                'plan_ledit', 'img_path_ledit', 'dose_path_ledit',
                'init_fluence_ledit', 'lower_var_ledit', 'upper_var_ledit',
                'ref_vol_ledit', 'ref_dose_ledit', 'baseline_ledit'))

            # Loop over the composer subwidgets
            for i in range(3):

                # Enable the subwidget
                self.composer_widget.widget(i).setEnabled(True)

            # Enable some fields
            self.set_enabled((
                'save_pbutton', 'drop_pbutton', 'update_configuration_pbutton',
                'reset_configuration_pbutton', 'update_optimization_pbutton',
                'reset_optimization_pbutton', 'update_evaluation_pbutton',
                'reset_evaluation_pbutton', 'configure_pbutton',
                'visualize_pbutton', 'actions_show_config_tbutton',
                'actions_show_datahub_tbutton', 'actions_show_log_tbutton',
                'actions_export_to_pyfile_tbutton',
                'compare_show_config_tbutton', 'compare_show_datahub_tbutton',
                'compare_show_log_tbutton', 'compare_export_to_pyfile_tbutton')
                )

            # Disable some fields
            self.set_disabled((
                'model_pbutton', 'optimize_pbutton', 'evaluate_pbutton'))

            # Set the status bar to configuration-ready
            self.status_bar.showMessage("Ready for configuration ...")

            # Get the treatment plan instance
            instance = self.plans[selection]

            # Check if the instance has already been configured
            if (all(getattr(instance, unit) is not None for unit in (
                   'input_checker', 'patient_loader', 'plan_generator',
                   'dose_info_generator'))
                    and instance.datahub.state >= 1):

                # Add the CT cube to the slice widget
                self.slice_widget.add_ct()

                # Adjust the slider
                self.adjust_slider_by_orientation()

                # Update the slice widget images
                self.slice_widget.update_images()

                # Get the segmentation dictionary from the instance
                segmentation = instance.datahub.segmentation

                # Get the machine learning components
                ml_components = (
                    get_machine_learning_constraints(segmentation)
                    + get_machine_learning_objectives(segmentation))

                # Check if any machine learning components are present
                if len(ml_components) > 0:

                    # Enable the modeling button
                    self.model_pbutton.setEnabled(True)

                    # Set the status bar to modeling-ready
                    self.status_bar.showMessage("Ready for modeling ...")

                    # Check if any component has not been modeled yet
                    if (any(getattr(component, unit) is None
                            for unit in ('data_model_handler', 'model')
                            for component in ml_components)
                            or instance.datahub.state == 1):

                        return

                # Enable the optimization button
                self.optimize_pbutton.setEnabled(True)

                # Set the status bar to optimization-ready
                self.status_bar.showMessage("Ready for optimization ...")

                # Check if the instance has already been optimized
                if (getattr(instance, 'fluence_optimizer') is not None
                        and 'optimized_dose' in instance.datahub.optimization
                        and instance.datahub.state >= 3):

                    # Add the dose cube to the slice widget
                    self.slice_widget.add_dose()

                    # Update the slice widget images
                    self.slice_widget.update_images()

                    # Enable the evaluation button
                    self.evaluate_pbutton.setEnabled(True)

                    # Set the status bar to evaluation-ready
                    self.status_bar.showMessage("Ready for evaluation ...")

                # Check if the instance has already been evaluated
                if (all(getattr(instance, unit) is not None for unit in (
                        'dose_histogram', 'dosimetrics'))
                        and instance.datahub.state == 4):

                    # Add style and input data to the DVH widget
                    self.dvh_widget.add_style_and_data(
                        instance.datahub.dose_histogram)

                    # Update the DVH plot
                    self.dvh_widget.update_dvh()

                    # Set the status bar to plan-ready
                    self.status_bar.showMessage(
                        f'"{self.plan_ledit.text()}" plan is ready ...')

        else:

            # Reset the last selection
            self.last_selection = ''

            # Clear the composer tabs
            self.clear_configuration()
            self.clear_optimization()
            self.clear_evaluation()

            # Loop over the composer subwidgets
            for i in range(3):

                # Disable the subwidget
                self.composer_widget.widget(i).setEnabled(False)

            # Disable some fields
            self.set_disabled((
                'save_pbutton', 'drop_pbutton', 'update_configuration_pbutton',
                'reset_configuration_pbutton', 'components_minus_tbutton',
                'components_edit_tbutton', 'init_fluence_ledit',
                'init_fluence_tbutton', 'ref_plan_cbox',
                'update_optimization_pbutton', 'reset_optimization_pbutton',
                'update_evaluation_pbutton', 'reset_evaluation_pbutton',
                'configure_pbutton', 'model_pbutton', 'optimize_pbutton',
                'evaluate_pbutton', 'visualize_pbutton',
                'actions_show_config_tbutton', 'actions_show_datahub_tbutton',
                'actions_show_log_tbutton', 'actions_export_to_pyfile_tbutton',
                'reference_cbox', 'compare_pbutton',
                'compare_show_config_tbutton', 'compare_show_datahub_tbutton',
                'compare_show_log_tbutton', 'compare_export_to_pyfile_tbutton',
                'plane_cbox', 'opacity_sbox', 'slice_selection_sbar',
                'stop_thread_pbutton'))

            # Clear the reference combo box for comparison
            self.reference_cbox.clear()

            # Reset the slice widget
            self.slice_widget.reset_images()

            # Reset the DVH widget
            self.dvh_widget.reset_dvh()

            # Reset the status bar to the initial message
            self.status_bar.showMessage(
                "Ready to load/select/create a treatment plan ...")

    def open_plan_creation_window(self):
        """Open the plan creation window."""

        # Get the scroll bar of the window
        scroll_creator = self.plan_creation_window.scroll_creator

        # Set the vertical scroll bar position
        scroll_creator.verticalScrollBar().setValue(
            scroll_creator.verticalScrollBar().minimum())

        # Loop over the input fields
        for widget in (
                'plan_ledit', 'ref_plan_cbox', 'img_path_ledit',
                'dose_path_ledit', 'dose_res_ledit_x', 'dose_res_ledit_y',
                'dose_res_ledit_z', 'components_lwidget'):

            # Clear the field
            getattr(self.plan_creation_window, widget).clear()

        # Check if any plans are available
        if len(self.plans) > 0:

            # Loop over the plans
            for plan in self.plans:

                # Add an item to the reference combo box
                self.plan_creation_window.ref_plan_cbox.addItem(plan)

            # Enable the reference combo box
            self.plan_creation_window.ref_plan_cbox.setEnabled(True)

        else:

            # Disable the reference combo box
            self.plan_creation_window.ref_plan_cbox.setEnabled(False)

        # Sort the items in the reference combo box alphabetically
        self.plan_creation_window.ref_plan_cbox.model().sort(0)

        # Insert the 'None' item
        self.plan_creation_window.ref_plan_cbox.insertItem(0, 'None')

        # Set the reference combo box to the 'None' item
        self.plan_creation_window.ref_plan_cbox.setCurrentIndex(0)

        # Set the position of the window
        self.plan_creation_window.position()

        # Show the window
        self.plan_creation_window.show()

    def open_settings_window(self):
        """Open the settings window."""

        # Set the current display resolution
        self.settings_window.resolution_cbox.setItemText(
            0, 'x'.join(map(str, (self.width(), self.height()))))

        # Set the resolution combo box to the current value
        self.settings_window.resolution_cbox.setCurrentIndex(0)

        # Set the position of the window
        self.settings_window.position()

        # Show the window
        self.settings_window.show()

    def open_info_window(self):
        """Open the information window."""

        # Set the position of the window
        self.info_window.position()

        # Show the window
        self.info_window.show()

    def exit_window(self):
        """Exit the session and close the window."""

        # Close the window
        self.close()

    def open_question_dialog(self):
        """Open a question dialog."""

        # Initialize the dialog source dictionary
        sources = {
            'drop_pbutton': (
                "Dropping the current treatment plan will irreversibly remove "
                "it from the GUI and from the datahub. Are you sure you want "
                "to proceed?",
                self.drop_tpi),
            'exit_pbutton': (
                "Do you really want to close the pyanno4rt GUI?",
                self.exit_window),
            'update_configuration_pbutton': (
                "Updating will change the configuration parameters of the "
                "current treatment plan and reset it to the initialization "
                "state. Are you sure you want to proceed?",
                self.update_configuration),
            'reset_configuration_pbutton': (
                "Resetting will change all configuration tab fields to the "
                "current state of the treatment plan. Are you sure you want "
                "to proceed?",
                self.set_configuration),
            'clear_configuration_pbutton': (
                "Clearing will change all configuration tab fields to the "
                "default state. Are you sure you want to proceed?",
                self.clear_configuration),
            'update_optimization_pbutton': (
                "Updating will change the optimization parameters of the "
                "current treatment plan and reset it to the modeling or "
                "optimization state. Are you sure you want to proceed?",
                self.update_optimization),
            'reset_optimization_pbutton': (
                "Resetting will change all optimization tab fields to the "
                "current state of the treatment plan. Are you sure you want "
                "to proceed?",
                self.set_optimization),
            'clear_optimization_pbutton': (
                "Clearing will change all optimization tab fields to the "
                "default state. Are you sure you want to proceed?",
                self.clear_optimization),
            'update_evaluation_pbutton': (
                "Updating will change the evaluation parameters of the "
                "current treatment plan and reset it to the optimization "
                "state. Are you sure you want to proceed?",
                self.update_evaluation),
            'reset_evaluation_pbutton': (
                "Resetting will change all evaluation tab fields to the "
                "current state of the treatment plan. Are you sure you want "
                "to proceed?",
                self.set_evaluation),
            'clear_evaluation_pbutton': (
                "Clearing will change all evaluation tab fields to the "
                "default state. Are you sure you want to proceed?",
                self.clear_evaluation)}

        # Get the message and function call from the sender name
        message, call = sources[self.sender().objectName()]

        # Check if the question dialog is confirmed
        if (QMessageBox.question(self, 'pyanno4rt', message)
                == QMessageBox.Yes):

            # Call the source function
            call()

    def add_imaging_path(self):
        """Add the CT and segmentation data from a folder."""

        # Get the file path
        path, _ = QFileDialog.getOpenFileName(
            self, 'Select a patient data file', '',
            'CT/Segmentation data (*.dcm *.mat *.p)')

        # Check if the file path exists
        if path:

            # Check if a DICOM file is selected
            if path.endswith('.dcm'):

                # Get the directory path
                path = dirname(path)

            # Set the imaging path field
            self.img_path_ledit.setText(abspath(path))

            # Set the imaging path field cursor position to zero
            self.img_path_ledit.setCursorPosition(0)

    def add_dose_matrix_path(self):
        """Add the dose-influence matrix from a folder."""

        # Get the file path
        path, _ = QFileDialog.getOpenFileName(
            self, 'Select a dose-influence matrix file', '',
            'Dose-influence matrix (*.mat *.npy)')

        # Check if the file path exists
        if path:

            # Set the dose matrix path field
            self.dose_path_ledit.setText(abspath(path))

            # Set the dose matrix path field cursor position to zero
            self.dose_path_ledit.setCursorPosition(0)

    def update_configuration(self):
        """Update the configuration parameters."""

        # Get the treatment plan instance
        instance = self.plans[self.plan_ledit.text()]

        try:

            # Update the configuration dictionary
            instance.update(self.transform_configuration_to_dict())

            # Load the segment names and types
            self.segments = load_segments_from_path(self.img_path_ledit.text())

        except Exception as exception:

            # Show a warning message box
            QMessageBox.warning(
                self, "pyanno4rt",
                "Exception occurred during configuration dictionary update - "
                "please check the inputs! \n\n"
                f"{type(exception).__name__}: {str(exception)}")

            # Raise the exception
            raise exception

        # Reset the datahub state
        instance.datahub.state = 0

        # Clear the display segments
        self.display_segments_cbox.clear()

        # Add the segment items to the display segments
        self.display_segments_cbox.addItems(list(self.segments.keys()))

        # Loop over the display segment items
        for item in (
                self.display_segments_cbox.model().item(index)
                for index in range(self.display_segments_cbox.count())):

            # Set the item to checked or unchecked
            item.setCheckState(2*(
                item.text() in instance.evaluation['display_segments'] or
                instance.evaluation['display_segments'] == []))

        # Reset the slice widget
        self.slice_widget.reset_images()

        # Reset the DVH widget
        self.dvh_widget.reset_dvh()

        # Set the line edit cursor positions to zero
        self.set_zero_line_cursor((
            'plan_ledit', 'img_path_ledit', 'dose_path_ledit'))

        # Enable the configuration button
        self.configure_pbutton.setEnabled(True)

        # Disable some fields
        self.set_disabled(('optimize_pbutton', 'evaluate_pbutton'))

        # Set the status bar to configuration-ready
        self.status_bar.showMessage("Ready for configuration ...")

    def set_configuration(self):
        """Set the configuration parameters."""

        # Get the configuration dictionary
        configuration = self.plans[self.plan_ledit.text()].configuration

        # Set the treatment plan label
        self.plan_ledit.setText(configuration['label'])

        # Set the minimum logging level
        self.log_level_cbox.setCurrentText(configuration['min_log_level'])

        # Set the treatment modality
        self.modality_cbox.setCurrentText(configuration['modality'])

        # Set the number of fractions
        self.nfx_sbox.setValue(configuration['number_of_fractions'])

        # Set the imaging path
        self.img_path_ledit.setText(abspath(configuration['imaging_path']))

        # Load the segment names and types
        self.segments = load_segments_from_path(self.img_path_ledit.text())

        # Set the dose matrix path
        self.dose_path_ledit.setText(
            abspath(configuration['dose_matrix_path']))

        # Set the dose resolution
        self.dose_res_ledit_x.setText(str(configuration['dose_resolution'][0]))
        self.dose_res_ledit_y.setText(str(configuration['dose_resolution'][1]))
        self.dose_res_ledit_z.setText(str(configuration['dose_resolution'][2]))

        # Set the line edit cursor positions to zero
        self.set_zero_line_cursor((
            'plan_ledit', 'img_path_ledit', 'dose_path_ledit'))

    def clear_configuration(self):
        """Clear the configuration parameters."""

        # Check if the treatment plan label is not included in the selector
        if self.plan_ledit.text() not in (
                self.plan_select_cbox.itemText(i)
                for i in range(self.plan_select_cbox.count())):

            # Reset the treatment plan label
            self.plan_ledit.setText(self.base_configuration['label'])

        # Reset the minimum logging level
        self.log_level_cbox.setCurrentText(
            self.base_configuration['min_log_level'])

        # Reset the treatment modality
        self.modality_cbox.setCurrentText(self.base_configuration['modality'])

        # Reset the number of fractions
        self.nfx_sbox.setValue(self.base_configuration['number_of_fractions'])

        # Reset the imaging path
        self.img_path_ledit.setText(self.base_configuration['imaging_path'])

        # Reset the dose matrix path
        self.dose_path_ledit.setText(
            self.base_configuration['dose_matrix_path'])

        # Reset the dose resolution
        self.dose_res_ledit_x.setText(
            None if not self.base_configuration['dose_resolution']
            else self.base_configuration['dose_resolution'][0])
        self.dose_res_ledit_y.setText(
            None if not self.base_configuration['dose_resolution']
            else self.base_configuration['dose_resolution'][1])
        self.dose_res_ledit_z.setText(
            None if not self.base_configuration['dose_resolution']
            else self.base_configuration['dose_resolution'][2])

    def transform_configuration_to_dict(self):
        """
        Transform the configuration fields into a dictionary.

        Returns
        -------
        dict
            Dictionary with the configuration parameters.
        """

        # Create the configuration dictionary
        configuration = {
            'label': (
                None if not self.plan_ledit.text()
                else self.plan_ledit.text()),
            'min_log_level': self.log_level_cbox.currentText(),
            'modality': self.modality_cbox.currentText(),
            'number_of_fractions': self.nfx_sbox.value(),
            'imaging_path': (
                None if not self.img_path_ledit.text()
                else abspath(self.img_path_ledit.text())),
            'dose_matrix_path': (
                None if not self.dose_path_ledit.text()
                else abspath(self.dose_path_ledit.text())),
            'dose_resolution': (
                None if any(resolution == '' for resolution in (
                    self.dose_res_ledit_x.text(), self.dose_res_ledit_y.text(),
                    self.dose_res_ledit_z.text()))
                else [string_to_numeric(resolution) for resolution in (
                    self.dose_res_ledit_x.text(), self.dose_res_ledit_y.text(),
                    self.dose_res_ledit_z.text())]),
            }

        return configuration

    def add_dropdown_to_components(self):
        """Add the dropdown menu to the components 'plus' button."""

        # Initialize the dropdown menu
        menu = QMenu()

        # Add submenus for the different component types
        conv_menu = menu.addMenu('Conventional')
        rb_menu = menu.addMenu('Radiobiological')
        ml_menu = menu.addMenu('Machine Learning')

        # Loop over the component map items
        for label, component in component_map.items():

            # Check if the component is of conventional type
            if issubclass(component, ConventionalComponent):

                # Add the action to the conventional submenu
                conv_menu.addAction(
                    label, partial(self.open_component_window, label))

            # Check if the component is of machine learning type
            elif issubclass(component, MachineLearningComponent):

                # Add the action to the machine learning menu
                ml_menu.addAction(
                    label, partial(self.open_component_window, label))

            # Check if the component is of radiobiological type
            elif issubclass(component, RadiobiologicalComponent):

                # Add the action to the radiobiological menu
                rb_menu.addAction(
                    label, partial(self.open_component_window, label))

        # Set the popup mode for the component 'plus' button
        self.components_plus_tbutton.setPopupMode(2)

        # Add the dropdown menu to the 'plus' button
        self.components_plus_tbutton.setMenu(menu)

    def open_component_window(
            self,
            name):
        """
        Open a component window.

        Parameters
        ----------
        name : str
            Name of the component.
        """

        # Get the component window
        self.current_component_window = component_window_map[name](self)

        # Set the position of the window
        self.current_component_window.position()

        # Show the window
        self.current_component_window.show()

    def remove_component(self):
        """Remove the selected component."""

        # Remove the component from the plan component dictionary
        del self.plan_components[self.plan_ledit.text()][
            self.components_lwidget.currentItem().text()]

        # Remove the component item from the list widget
        self.components_lwidget.takeItem(self.components_lwidget.currentRow())

        # Clear the selection in the list widget
        self.components_lwidget.selectionModel().clear()

        # Disable some fields
        self.set_disabled((
            'components_minus_tbutton', 'components_edit_tbutton'))

    def edit_component(self):
        """Edit the selected component."""

        # Get the component
        component = self.plan_components[self.plan_ledit.text()][
            self.components_lwidget.currentItem().text()]

        # Get the component window
        self.current_component_window = component_window_map[
            component[next(iter(component))]['instance']['function']](self)

        # Load the component into the window
        self.current_component_window.load(component, edit=True)

        # Set the position of the window
        self.current_component_window.position()

        # Show the window
        self.current_component_window.show()

    def update_by_method(self):
        """Update the GUI by the optimization method."""

        # Clear the solver combo box
        self.solver_cbox.clear()

        # Check if the method is 'lexicographic'
        if self.method_cbox.currentText() == 'lexicographic':

            # Add the solver items
            self.solver_cbox.addItems(['scipy'])

            # Set the default item
            self.solver_cbox.setCurrentText('scipy')

        # Else, check if the method is 'pareto'
        elif self.method_cbox.currentText() == 'pareto':

            # Add the solver items
            self.solver_cbox.addItems(['pymoo'])

            # Set the default item
            self.solver_cbox.setCurrentText('pymoo')

        # Else, check if the method is 'weighted-sum'
        elif self.method_cbox.currentText() == 'weighted-sum':

            # Add the solver items
            self.solver_cbox.addItems(['ipyopt', 'proxmin', 'pypop7', 'scipy'])

            # Set the default item
            self.solver_cbox.setCurrentText('scipy')

    def update_by_solver(self):
        """Update the GUI by the solver."""

        # Clear the algorithm combo box
        self.algorithm_cbox.clear()

        # Check if the solver is 'ipyopt'
        if self.solver_cbox.currentText() == 'ipyopt':

            # Add the algorithm items
            self.algorithm_cbox.addItems(['mumps'])

            # Set the default item
            self.algorithm_cbox.setCurrentText('mumps')

        # Check if the solver is 'proxmin'
        elif self.solver_cbox.currentText() == 'proxmin':

            # Add the algorithm items
            self.algorithm_cbox.addItems(['admm', 'pgm', 'sdmm'])

            # Set the default item
            self.algorithm_cbox.setCurrentText('pgm')

        # Else, check if the solver is 'pymoo'
        elif self.solver_cbox.currentText() == 'pymoo':

            # Add the algorithm items
            self.algorithm_cbox.addItems(['NSGA3'])

            # Set the default item
            self.algorithm_cbox.setCurrentText('NSGA3')

        # Else, check if the solver is 'pypop7'
        elif self.solver_cbox.currentText() == 'pypop7':

            # Add the algorithm items
            self.algorithm_cbox.addItems(['LMCMA', 'LMMAES'])

            # Set the default item
            self.algorithm_cbox.setCurrentText('LMMAES')

        # Else, check if the solver is 'scipy'
        elif self.solver_cbox.currentText() == 'scipy':

            # Check if the method is 'lexicographic'
            if self.method_cbox.currentText() == 'lexicographic':

                # Add the algorithm items
                self.algorithm_cbox.addItems(['trust-constr'])

                # Set the default item
                self.algorithm_cbox.setCurrentText('trust-constr')

            else:

                # Add the algorithm items
                self.algorithm_cbox.addItems(
                    ['L-BFGS-B', 'TNC', 'trust-constr'])

                # Set the default item
                self.algorithm_cbox.setCurrentText('L-BFGS-B')

    def update_by_initial_strategy(self):
        """Update the GUI by the initial strategy."""

        # Check if the initial strategy is different from 'warm-start'
        if self.init_strat_cbox.currentText() != 'warm-start':

            # Disable some fields
            self.set_disabled((
                'init_fluence_ledit', 'init_fluence_tbutton', 'ref_plan_cbox'))

        else:

            # Enable some fields
            self.set_enabled((
                'init_fluence_ledit', 'init_fluence_tbutton', 'ref_plan_cbox'))

    def update_by_initial_fluence(self):
        """Update the GUI by the initial fluence vector."""

        # Enable or disable the reference combo box
        self.ref_plan_cbox.setEnabled(self.init_fluence_ledit.text() == '')

    def add_initial_fluence_vector(self):
        """Add the initial fluence vector from a file."""

        # Get the file path
        path, _ = QFileDialog.getOpenFileName(
            self, 'Select an initial fluence vector file', '',
            'Fluence vector (*.json *.p *.txt)')

        # Check if the file path exists
        if path:

            # Set the initial fluence field
            self.init_fluence_ledit.setText(str(load_list_from_file(path)))

            # Set the initial fluence field cursor position to zero
            self.init_fluence_ledit.setCursorPosition(0)

    def update_reference_plans(self):
        """Update the reference plans."""

        # Clear the reference combo box
        self.ref_plan_cbox.clear()

        # Get the list of reference plans
        reference_plans = [
            label for label, instance in self.plans.items()
            if label != self.plan_select_cbox.currentText()
            and instance.datahub.state >= 3]

        # Add the reference plans to the combo box
        self.ref_plan_cbox.addItems([str(None)] + reference_plans)

        # Clear the reference combo box for comparison
        self.reference_cbox.clear()

        # Add the plans for comparison
        self.reference_cbox.addItems(reference_plans)

        # Enable or disable the combo box
        self.reference_cbox.setEnabled(len(reference_plans) > 0)

        # Enable or disable the comparison button
        self.compare_pbutton.setEnabled(len(reference_plans) > 0)

    def update_by_reference(self):
        """Update the GUI by the reference plan."""

        # Check if a reference plan has been passed
        if self.ref_plan_cbox.currentText() != 'None':

            # Disable some fields
            self.set_disabled(('init_fluence_ledit', 'init_fluence_tbutton'))

        else:

            # Enable some fields
            self.set_enabled(('init_fluence_ledit', 'init_fluence_tbutton'))

    def add_lower_var_bounds(self):
        """Add the lower variable bounds from a file."""

        # Get the file path
        path, _ = QFileDialog.getOpenFileName(
            self, 'Select a lower variable bounds file', '',
            'Fluence vector (*.json *.p *.txt)')

        # Check if the file path exists
        if path:

            # Set the lower variable bounds field
            self.lower_var_ledit.setText(str(load_list_from_file(path)))

            # Set the lower variable bounds field cursor position to zero
            self.lower_var_ledit.setCursorPosition(0)

    def add_upper_var_bounds(self):
        """Add the upper variable bounds from a file."""

        # Get the file path
        path, _ = QFileDialog.getOpenFileName(
            self, 'Select an upper variable bounds file', '',
            'Fluence vector (*.json *.p *.txt)')

        # Check if the file path exists
        if path:

            # Set the upper variable bounds field
            self.upper_var_ledit.setText(str(load_list_from_file(path)))

            # Set the upper variable bounds field cursor position to zero
            self.upper_var_ledit.setCursorPosition(0)

    def update_optimization(self):
        """Update the optimization parameters."""

        # Get the treatment plan instance
        instance = self.plans[self.plan_ledit.text()]

        try:

            # Overwrite the optimization dictionary
            instance.update(self.transform_optimization_to_dict())

        except Exception as exception:

            # Show a warning message box
            QMessageBox.warning(
                self, "pyanno4rt",
                "Exception occurred during optimization dictionary update - "
                "please check the inputs! \n\n"
                f"{type(exception).__name__}: {str(exception)}")

            # Raise the exception
            raise exception

        # Check if the the plan generator has been initialized
        if instance.plan_generator is not None:

            # Overwrite the components in the plan generator
            instance.plan_generator.components = instance.optimization[
                'components']

            # Update the components in the datahub
            instance.plan_generator.set_optimization_components(
                verbose=False)

        # Reset the slice widget images
        self.slice_widget.reset_images()

        # Reset the DVH widget
        self.dvh_widget.reset_dvh()

        # Set the line edit cursor positions to zero
        self.set_zero_line_cursor((
            'init_fluence_ledit', 'lower_var_ledit', 'upper_var_ledit'))

        # Disable the evaluation button
        self.evaluate_pbutton.setEnabled(False)

        # Check if the instance has already been configured
        if all(getattr(instance, unit) is not None for unit in (
                'input_checker', 'patient_loader', 'plan_generator',
                'dose_info_generator')):

            # Reset the datahub state
            instance.datahub.state = 1

            # Add the CT cube to the slice widget
            self.slice_widget.add_ct()

            # Adjust the slider
            self.adjust_slider_by_orientation()

            # Update the slice widget images
            self.slice_widget.update_images()

            # Get the segmentation dictionary from the instance
            segmentation = instance.datahub.segmentation

            # Get the machine learning components
            ml_components = (
                get_machine_learning_constraints(segmentation)
                + get_machine_learning_objectives(segmentation))

            # Check if any machine learning components are present
            if len(ml_components) > 0:

                # Enable the modeling button
                self.model_pbutton.setEnabled(True)

                # Disable the optimization button
                self.optimize_pbutton.setEnabled(False)

                # Set the status bar to modeling-ready
                self.status_bar.showMessage("Ready for modeling ...")

            else:

                # Disable the modeling button
                self.model_pbutton.setEnabled(False)

                # Enable the optimization button
                self.optimize_pbutton.setEnabled(True)

                # Set the status bar to optimization-ready
                self.status_bar.showMessage("Ready for optimization ...")

    def set_optimization(self):
        """Set the optimization parameters."""

        def set_single_component(segment, component):
            """Set a single component."""

            # Map the component and segment type to the icon paths
            paths = {
                'objective_TARGET': (
                    ":/special_icons/icons_special/"
                    "target-red-svgrepo-com.svg"),
                'objective_OAR': (
                    ":/special_icons/icons_special/"
                    "target-green-svgrepo-com.svg"),
                'constraint_TARGET': (
                    ":/special_icons/icons_special/"
                    "frame-red-svgrepo-com.svg"),
                'constraint_OAR': (
                    ":/special_icons/icons_special/"
                    "frame-green-svgrepo-com.svg")
                }

            # Get the icon path
            icon_path = paths[f'{component["type"]}_{self.segments[segment]}']

            # Initialize the icon object
            icon = QIcon()

            # Add the pixmap to the icon
            icon.addPixmap(QPixmap(icon_path), QIcon.Normal, QIcon.Off)

            # Get the component parameters
            parameters = component['instance']['parameters']

            # Check if the identifier parameter has been specified
            if 'identifier' in parameters:

                # Get the identifier string
                identifier = parameters['identifier']

            else:

                # Set the identifier to None
                identifier = None

            # Check if the embedding parameter is specified
            if 'embedding' in parameters:

                # Get the embedding string
                embedding = f'embedding: {str(parameters["embedding"])}'

            else:

                # Set the embedding string to the default
                embedding = 'embedding: active'

            # Check if the weight parameter has been specified
            if 'weight' in parameters:

                # Get the weight string
                weight = f'weight: {str(float(parameters["weight"]))}'

            else:

                # Set the weight string to the default
                weight = 'weight: 1'

            # Join the parameter strings
            component_string = ' - '.join((string for string in (
                segment, component['instance']['function'], identifier,
                embedding, weight) if string))

            # Add the icon with the component string to the list
            self.components_lwidget.addItem(
                QListWidgetItem(icon, component_string))

            # Add the component to the plan component dictionary
            self.plan_components[self.plan_ledit.text()][component_string] = {
                segment: component}

        # Get the optimization dictionary
        optimization = self.plans[self.plan_ledit.text()].optimization

        # Clear the component list
        self.components_lwidget.clear()

        # Loop over the components
        for segment, component in optimization['components'].items():

            # Check if the component is a list
            if isinstance(component, list):

                # Loop over the component elements
                for element in component:

                    # Set the element
                    set_single_component(segment, element)

            else:

                # Set the component
                set_single_component(segment, component)

        # Set the optimization method
        self.method_cbox.setCurrentText(optimization['method'])

        # Set the solver
        self.solver_cbox.setCurrentText(optimization['solver'])

        # Set the algorithm
        self.algorithm_cbox.setCurrentText(optimization['algorithm'])

        # Set the initialization strategy
        self.init_strat_cbox.setCurrentText(optimization['initial_strategy'])

        # Check if the initialization strategy is different from 'warm-start'
        if self.init_strat_cbox.currentText() != 'warm-start':

            # Disable some fields
            self.set_disabled((
                'init_fluence_ledit', 'init_fluence_tbutton', 'ref_plan_cbox'))

        else:

            # Enable some fields
            self.set_enabled((
                'init_fluence_ledit', 'init_fluence_tbutton', 'ref_plan_cbox'))

        # Check if an initial fluence vector has been specified
        if optimization['initial_fluence_vector']:

            # Set the initial fluence vector
            self.init_fluence_ledit.setText(
                str(optimization['initial_fluence_vector'])[1:-1])

        else:

            # Clear the initial fluence vector
            self.init_fluence_ledit.clear()

        # Set the initial reference plan
        self.ref_plan_cbox.setCurrentIndex(0)

        # Check if the lower variable bounds are different from zero
        if optimization['lower_variable_bounds'] != 0:

            # Set the lower variable bounds
            self.lower_var_ledit.setText(
                str(optimization['lower_variable_bounds']))

        else:

            # Clear the lower variable bounds
            self.lower_var_ledit.clear()

        # Check if the upper variable bounds have been specified
        if optimization['upper_variable_bounds']:

            # Set the upper variable bounds
            self.upper_var_ledit.setText(
                str(optimization['upper_variable_bounds']))

        else:

            # Clear the upper variable bounds
            self.upper_var_ledit.clear()

        # Set the maximum number of iterations
        self.max_iter_sbox.setValue(optimization['max_iter'])

        # Set the tolerance
        self.tolerance_ledit.setText(
            '' if optimization['tolerance'] == 0.001
            else str(optimization['tolerance']))

        # Set the line edit cursor positions to zero
        self.set_zero_line_cursor((
            'init_fluence_ledit', 'lower_var_ledit', 'upper_var_ledit'))

    def clear_optimization(self):
        """Clear the optimization parameters."""

        # Reset the components
        self.components_lwidget.clear()

        # Reset the optimization method
        self.method_cbox.setCurrentText(self.base_optimization['method'])

        # Reset the solver
        self.solver_cbox.setCurrentText(self.base_optimization['solver'])

        # Reset the algorithm
        self.algorithm_cbox.setCurrentText(self.base_optimization['algorithm'])

        # Reset the initialization strategy
        self.init_strat_cbox.setCurrentText(
            self.base_optimization['initial_strategy'])

        # Reset the initial fluence vector
        self.init_fluence_ledit.clear()

        # Reset the initial reference plan
        self.ref_plan_cbox.setCurrentIndex(0)

        # Reset the lower variable bounds
        self.lower_var_ledit.clear()

        # Reset the upper variable bounds
        self.upper_var_ledit.clear()

        # Reset the maximum number of iterations
        self.max_iter_sbox.setValue(self.base_optimization['max_iter'])

        # Reset the tolerance
        self.tolerance_ledit.setText(
            '' if self.base_optimization['tolerance'] == 0.001
            else str(self.base_optimization['tolerance']))

    def transform_optimization_to_dict(self):
        """
        Transform the optimization fields into a dictionary.

        Returns
        -------
        dict
            Dictionary with the optimization parameters.
        """

        # Initialize the components dictionary
        components = {}

        # Loop over the treatment plan components
        for component in self.plan_components.get(
                self.plan_ledit.text(), {}).values():

            # Get the key and value of the component
            (key, value), = component.items()

            # Check if the key is not yet included in the dictionary
            if key not in components:

                # Enter the value into the dictionary
                components[key] = value

            else:

                # Extend the component to a list
                components[key] = [components[key], value]

        # Get the initial fluence sources
        sources = (
            self.init_fluence_ledit.text(), self.ref_plan_cbox.currentText())

        # Convert the lower variable bounds
        lower_variable_bounds = add_square_brackets(
            self.lower_var_ledit.text())

        # Convert the upper variable bounds
        upper_variable_bounds = add_square_brackets(
            self.upper_var_ledit.text())

        # Create the optimization dictionary from the input fields
        optimization = {
            'components': components,
            'method': self.method_cbox.currentText(),
            'solver': self.solver_cbox.currentText(),
            'algorithm': self.algorithm_cbox.currentText(),
            'initial_strategy': self.init_strat_cbox.currentText(),
            'initial_fluence_vector': (
                None if sources == ('', 'None')
                else loads(self.init_fluence_ledit.text()) if sources[0] != ''
                else self.plans[sources[1]].datahub.optimization[
                    'optimized_fluence'].tolist()),
            'lower_variable_bounds': (
                0 if not lower_variable_bounds
                else loads(lower_variable_bounds)),
            'upper_variable_bounds': (
                 None if not upper_variable_bounds
                 else loads(upper_variable_bounds)),
            'max_iter': self.max_iter_sbox.value(),
            'tolerance': (
                1e-3 if self.tolerance_ledit.text() == ''
                else loads(self.tolerance_ledit.text()))
            }

        return optimization

    def update_evaluation(self):
        """Update the evaluation parameters."""

        # Get the treatment plan instance
        instance = self.plans[self.plan_ledit.text()]

        try:

            # Update the evaluation dictionary
            instance.update(self.transform_evaluation_to_dict())

        except Exception as exception:

            # Show a warning message box
            QMessageBox.warning(
                self, "pyanno4rt",
                "Exception occurred during evaluation dictionary update - "
                "please check the inputs! \n\n"
                f"{type(exception).__name__}: {str(exception)}")

            # Raise the exception
            raise exception

        # Reset the DVH widget
        self.dvh_widget.reset_dvh()

        # Set the line edit cursor positions to zero
        self.set_zero_line_cursor(('ref_vol_ledit', 'ref_dose_ledit'))

        # Check if the instance has already been optimized
        if (getattr(instance, 'fluence_optimizer') is not None
                and 'optimized_dose' in instance.datahub.optimization):

            # Reset the datahub state
            instance.datahub.state = 3

            # Set the status bar to evaluation-ready
            self.status_bar.showMessage("Ready for evaluation ...")

    def set_evaluation(self):
        """Set the evaluation parameters."""

        # Get the evaluation dictionary
        evaluation = self.plans[self.plan_ledit.text()].evaluation

        # Set the DVH type
        self.dvh_type_cbox.setCurrentText(evaluation['dvh_type'])

        # Set the number of DVH points
        self.n_points_sbox.setValue(evaluation['number_of_points'])

        # Set the reference volume
        self.ref_vol_ledit.setText(
            '' if evaluation['reference_volume'] == [2, 5, 50, 95, 98]
            else str(evaluation['reference_volume'])[1:-1])

        # Set the reference dose values
        self.ref_dose_ledit.setText(
            '' if evaluation['reference_dose'] == []
            else str(evaluation['reference_dose'])[1:-1])

        # Clear the display segments
        self.display_segments_cbox.clear()

        # Add the segment items to the display segments
        self.display_segments_cbox.addItems(list(self.segments.keys()))

        # Loop over the display segment items
        for item in (
                self.display_segments_cbox.model().item(index)
                for index in range(self.display_segments_cbox.count())):

            # Set the item to checked or unchecked
            item.setCheckState(2*(
                item.text() in evaluation['display_segments'] or
                evaluation['display_segments'] == []))

        # Loop over the display metrics items
        for item in (
                self.display_metrics_cbox.model().item(index)
                for index in range(self.display_metrics_cbox.count())):

            # Set the item to checked or unchecked
            item.setCheckState(2*(
                item.text() in evaluation['display_metrics'] or
                evaluation['display_metrics'] == []))

        # Set the line edit cursor positions to zero
        self.set_zero_line_cursor(('ref_vol_ledit', 'ref_dose_ledit'))

    def clear_evaluation(self):
        """Clear the evaluation parameters."""

        # Reset the DVH type
        self.dvh_type_cbox.setCurrentText(self.base_evaluation['dvh_type'])

        # Reset the number of DVH points
        self.n_points_sbox.setValue(self.base_evaluation['number_of_points'])

        # Reset the reference volume
        self.ref_vol_ledit.setText(
            str(self.base_evaluation['reference_volume'])[1:-1])

        # Reset the reference dose values
        self.ref_dose_ledit.setText(
            str(self.base_evaluation['reference_dose'])[1:-1])

        # Loop over the display segments
        for index in range(self.display_segments_cbox.count()):

            # Reset the display segments to checked
            self.display_segments_cbox.model().item(index).setCheckState(2)

        # Loop over the display metrics
        for index in range(self.display_metrics_cbox.count()):

            # Reset the display metric to checked
            self.display_metrics_cbox.model().item(index).setCheckState(2)

    def transform_evaluation_to_dict(self):
        """
        Transform the evaluation fields into a dictionary.

        Returns
        -------
        dict
            Dictionary with the evaluation parameters.
        """

        # Convert the reference volume from the field
        reference_volume = add_square_brackets(self.ref_vol_ledit.text())

        # Convert the reference dose from the field
        reference_dose = add_square_brackets(self.ref_dose_ledit.text())

        # Create the evaluation dictionary from the input fields
        evaluation = {
            'dvh_type': self.dvh_type_cbox.currentText(),
            'number_of_points': self.n_points_sbox.value(),
            'reference_volume': (
                [2, 5, 50, 95, 98] if reference_volume == ''
                else loads(reference_volume)),
            'reference_dose': (
                [] if reference_dose == ''
                else loads(reference_dose)),
            'display_segments': self.display_segments_cbox.currentData(),
            'display_metrics': self.display_metrics_cbox.currentData()
            }

        return evaluation

    def run_stop_and_loader(
            self,
            turn_on):
        """
        Turn the stop button and the loader label on/off.

        Parameters
        ----------
        turn_on : bool
            Indicator for turning on/off the stop button and the loader label.
        """

        # Check if the elements should be turned on
        if turn_on:

            # Show the loader label
            self.loader_label.show()

            # Enable the stop button
            self.stop_thread_pbutton.setEnabled(True)

        else:

            # Hide the loader label
            self.loader_label.hide()

            # Disable the stop button
            self.stop_thread_pbutton.setEnabled(False)

    def configure_thread(
            self,
            task,
            fail_method,
            finish_method):
        """
        Configure the worker thread.

        Parameters
        ----------
        task : object of class :class:`~method`
            The task to be performed in the worker thread.

        fail_method : object of class :class:`~method`
            The method to be executed after failure of the task.

        finish_method : object of class :class:`~method`
            The method to be executed after success of the task.
        """

        # Initialize the thread
        self.thread = QThread()

        # Initialize the worker
        self.worker = Worker(task)

        # Move the worker to the thread
        self.worker.moveToThread(self.thread)

        # Connect the worker failed signals
        self.worker.failed.connect(fail_method)
        self.worker.failed.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.wait)
        self.worker.failed.connect(self.worker.deleteLater)

        # Connect the worker finished signals
        self.worker.finished.connect(finish_method)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.thread.wait)
        self.worker.finished.connect(self.worker.deleteLater)

        # Connect the thread signals
        self.thread.started.connect(self.worker.run)
        self.thread.finished.connect(self.thread.deleteLater)

        # Start the thread
        self.thread.start()

    def initialize(self):
        """Initialize the treatment plan."""

        # Turn the stop button and loader label on
        self.run_stop_and_loader(True)

        # Initialize the treatment plan
        treatment_plan = TreatmentPlan(
            self.transform_configuration_to_dict(),
            self.transform_optimization_to_dict(),
            self.transform_evaluation_to_dict())

        # Activate the treatment plan
        self.activate(treatment_plan)

        # Set the status bar to configuration-ready
        self.status_bar.showMessage("Ready for configuration ...")

        # Turn the stop button and loader label off
        self.run_stop_and_loader(False)

    def start_configure(self):
        """Start the configuration process."""

        # Turn the stop button and loader label on
        self.run_stop_and_loader(True)

        # Set up the configuration thread
        self.configure_thread(
            self.configure, self.update_after_configure_failure,
            self.update_after_configure_success)

    def configure(self):
        """Configure the treatment plan."""

        # Run the configuration method of the treatment plan
        self.plans[self.plan_ledit.text()].configure()

    def update_after_configure_failure(
            self,
            exception):
        """
        Update the GUI after failure of the configuration.

        Parameters
        ----------
        exception : object of class :class:`Exception`
            The exception raised from failure.
        """

        # Set the status bar to error
        self.status_bar.showMessage(
            "Exception occurred during configuration - please check the "
            "configuration parameters ...")

        # Turn the stop button and loader label off
        self.run_stop_and_loader(False)

        # Show a warning message box
        QMessageBox.warning(self, "pyanno4rt", str(exception))

        # Raise the exception
        raise exception

    def update_after_configure_success(self):
        """Update the GUI after success of the configuration."""

        # Reset the slice widget
        self.slice_widget.reset_images()

        # Add the CT cube to the slice widget
        self.slice_widget.add_ct()

        # Adjust the slider
        self.adjust_slider_by_orientation()

        # Update the slice widget images
        self.slice_widget.update_images()

        # Reset the DVH widget
        self.dvh_widget.reset_dvh()

        # Show the CT/Dose tab
        self.viewer_widget.setCurrentIndex(0)

        # Update the log output
        self.log_window.update_log_output()

        # Set the line edit cursor positions to zero
        self.set_zero_line_cursor((
            'plan_ledit', 'img_path_ledit', 'dose_path_ledit',
            'init_fluence_ledit', 'lower_var_ledit', 'upper_var_ledit',
            'ref_vol_ledit', 'ref_dose_ledit'))

        # Enable some fields
        self.set_enabled(('plane_cbox', 'slice_selection_sbar'))

        # Disable some fields
        self.set_disabled((
            'model_pbutton', 'optimize_pbutton', 'evaluate_pbutton'))

        # Get the segmentation dictionary
        segmentation = self.plans[self.plan_ledit.text()].datahub.segmentation

        # Check if any machine learning components are non-
        if len(get_machine_learning_constraints(segmentation)
               + get_machine_learning_objectives(segmentation)) > 0:

            # Enable the modeling button
            self.model_pbutton.setEnabled(True)

            # Set the status bar to modeling-ready
            self.status_bar.showMessage("Ready for modeling ...")

        else:

            # Enable the optimization button
            self.optimize_pbutton.setEnabled(True)

            # Set the status bar to optimization-ready
            self.status_bar.showMessage("Ready for optimization ...")

        # Turn the stop button and loader label off
        self.run_stop_and_loader(False)

    def start_model(self):
        """Start the modeling process."""

        # Turn the stop button and loader label on
        self.run_stop_and_loader(True)

        # Set up the modeling thread
        self.configure_thread(
            self.model, self.update_after_model_failure,
            self.update_after_model_success)

    def model(self):
        """Set up the machine learning outcome prediction models."""

        # Run the modeling method of the treatment plan
        self.plans[self.plan_ledit.text()].model()

    def update_after_model_failure(
            self,
            exception):
        """
        Update the GUI after failure of the outcome modeling.

        Parameters
        ----------
        exception : object of class :class:`Exception`
            The exception raised from failure.
        """

        # Set the status bar to error
        self.status_bar.showMessage(
            "Exception occurred during modeling - please check the "
            "modeling parameters ...")

        # Turn the stop button and loader label off
        self.run_stop_and_loader(False)

        # Show a warning message box
        QMessageBox.warning(self, "pyanno4rt", str(exception))

        # Raise the exception
        raise exception

    def update_after_model_success(self):
        """Update the GUI after success of the outcome modeling."""

        # Reset the slice widget
        self.slice_widget.reset_images()

        # Add the CT cube to the slice widget
        self.slice_widget.add_ct()

        # Adjust the slider
        self.adjust_slider_by_orientation()

        # Update the slice widget images
        self.slice_widget.update_images()

        # Reset the DVH widget
        self.dvh_widget.reset_dvh()

        # Update the log output
        self.log_window.update_log_output()

        # Enable the optimization button
        self.optimize_pbutton.setEnabled(True)

        # Disable the evaluation button
        self.evaluate_pbutton.setEnabled(False)

        # Set the status bar to optimization-ready
        self.status_bar.showMessage("Ready for optimization ...")

        # Turn the stop button and loader label off
        self.run_stop_and_loader(False)

    def start_optimize(self):
        """Start the optimization process."""

        # Turn the stop button and loader label on
        self.run_stop_and_loader(True)

        # Set up the optimization thread
        self.configure_thread(
            self.optimize, self.update_after_optimize_failure,
            self.update_after_optimize_success)

    def optimize(self):
        """Optimize the treatment plan."""

        # Run the optimization method of the treatment plan
        self.plans[self.plan_ledit.text()].optimize()

    def update_after_optimize_failure(
            self,
            exception):
        """
        Update the GUI after failure of the optimization.

        Parameters
        ----------
        exception : object of class :class:`Exception`
            The exception raised from failure.
        """

        # Set the status bar to error
        self.status_bar.showMessage(
            "Exception occurred during optimization - please check the "
            "optimization parameters ...")

        # Turn the stop button and loader label off
        self.run_stop_and_loader(False)

        # Show a warning message box
        QMessageBox.warning(self, "pyanno4rt", str(exception))

        # Raise the exception
        raise exception

    def update_after_optimize_success(self):
        """Update the GUI after success of the optimization."""

        # Check if the dose contours have already been computed
        if self.slice_widget.dose_contours:

            # Loop over the dose contours
            for contour in self.slice_widget.dose_contours:

                # Set the dose contour data to zeros
                contour.setData(zeros(self.slice_widget.dose_cube[
                    :, :, self.slice_widget.slice].shape))

        # Add the dose cube to the slice widget
        self.slice_widget.add_dose()

        # Update the slice widget images
        self.slice_widget.update_images()

        # Reset the DVH widget
        self.dvh_widget.reset_dvh()

        # Show the CT/Dose tab
        self.viewer_widget.setCurrentIndex(0)

        # Update the log output
        self.log_window.update_log_output()

        # Set the line edit cursor positions to zero
        self.set_zero_line_cursor((
            'plan_ledit', 'img_path_ledit', 'dose_path_ledit',
            'init_fluence_ledit', 'lower_var_ledit', 'upper_var_ledit',
            'ref_vol_ledit', 'ref_dose_ledit'))

        # Enable some fields
        self.set_enabled(('evaluate_pbutton', 'opacity_sbox'))

        # Set the status bar to evaluation-ready
        self.status_bar.showMessage("Ready for evaluation ...")

        # Turn the stop button and loader label off
        self.run_stop_and_loader(False)

    def start_evaluate(self):
        """Start the evaluation process."""

        # Turn the stop button and loader label on
        self.run_stop_and_loader(True)

        # Set up the evaluation thread
        self.configure_thread(
            self.evaluate, self.update_after_evaluate_failure,
            self.update_after_evaluate_success)

    def evaluate(self):
        """Evaluate the treatment plan."""

        # Run the evaluation method of the treatment plan
        self.plans[self.plan_ledit.text()].evaluate()

    def update_after_evaluate_failure(
            self,
            exception):
        """
        Update the GUI after failure of the evaluation.

        Parameters
        ----------
        exception : object of class :class:`Exception`
            The exception raised from failure.
        """

        # Set the status bar to error
        self.status_bar.showMessage(
            "Exception occurred during evaluation - please check the "
            "evaluation parameters ...")

        # Turn the stop button and loader label off
        self.run_stop_and_loader(False)

        # Show a warning message box
        QMessageBox.warning(self, "pyanno4rt", str(exception))

        # Raise the exception
        raise exception

    def update_after_evaluate_success(self):
        """Update the GUI after success of the evaluation."""

        # Reset the DVH widget
        self.dvh_widget.reset_dvh()

        # Add style and input data to the DVH widget
        self.dvh_widget.add_style_and_data(
            self.plans[self.plan_ledit.text()].datahub.dose_histogram)

        # Update the DVH plot
        self.dvh_widget.update_dvh()

        # Update the log output
        self.log_window.update_log_output()

        # Set the line edit cursor positions to zero
        self.set_zero_line_cursor((
            'plan_ledit', 'img_path_ledit', 'dose_path_ledit',
            'init_fluence_ledit', 'lower_var_ledit', 'upper_var_ledit',
            'ref_vol_ledit', 'ref_dose_ledit'))

        # Show the DVH tab
        self.viewer_widget.setCurrentIndex(1)

        # Set the status bar to plan-ready
        self.status_bar.showMessage(
            f'"{self.plan_ledit.text()}" plan is ready ...')

        # Turn the stop button and loader label off
        self.run_stop_and_loader(False)

    def visualize(self):
        """Visualize the treatment plan."""

        # Run the visualization method of the treatment plan
        self.plans[self.plan_ledit.text()].visualize(parent=self)

    def open_configuration_window(self):
        """Open the plan configuration window."""

        # Get the treatment plan instance
        instance = self.plans[self.plan_ledit.text()]

        # Loop over the configuration file paths
        for key in ('imaging_path', 'dose_matrix_path'):

            # Convert into an absolute path
            instance.configuration[key] = abspath(instance.configuration[key])

        # Clear the configuration window
        self.config_window.tree_widget.clear()

        # Create the configuration tree from the input dictionaries
        self.config_window.create_tree_from_dict(data={
            key: value for key, value in vars(instance).items()
            if isinstance(value, dict)},
            parent=self.config_window.tree_widget)

        # Set the resize mode for the first tree column
        self.config_window.tree_widget.header().setSectionResizeMode(
            0, QHeaderView.Stretch)

        # Set the position of the window
        self.config_window.position()

        # Show the window
        self.config_window.show()

    def open_datahub_window(self):
        """Open the datahub content window."""

        # Get the treatment plan instance
        instance = self.plans[self.plan_ledit.text()]

        # Clear the datahub window
        self.datahub_window.tree_widget.clear()

        # Create the datahub tree from the internal plan dictionaries
        self.datahub_window.create_tree_from_dict(data={
            key: value for key, value in vars(instance.datahub).items()
            if isinstance(value, dict)},
            parent=self.datahub_window.tree_widget)

        # Set the resize mode for the first tree column
        self.datahub_window.tree_widget.header().setSectionResizeMode(
            0, QHeaderView.Stretch)

        # Set the position of the window
        self.datahub_window.position()

        # Show the window
        self.datahub_window.show()

    def open_log_window(self):
        """Open the log window."""

        # Set the position of the window
        self.log_window.position()

        # Show the window
        self.log_window.show()

    def export_to_pyfile(self):
        """Export the treatment plan instance to a Python file (.py)."""

        # Get the file path
        path, _ = QFileDialog.getSaveFileName(
            self, 'Export to Python file', 'tp.py', 'Python file (*.py)')

        # Check if the file path exists
        if path:

            # Get the treatment plan instance
            instance = self.plans[self.plan_ledit.text()]

            # Open a file stream
            with open(path, 'w', encoding='utf-8') as file:

                # Create a string replacement mapping
                mapping = {
                    '\n': '\n' + '    ',
                    'null': 'None',
                    'true': 'True',
                    'false': 'False'}

                # Convert the input dictionaries to formatted strings
                configuration, optimization, evaluation = (
                    reduce(lambda x, y: x.replace(*y),
                           [dumps(getattr(instance, dct), indent=4),
                            *list(mapping.items())])
                    for dct in ('configuration', 'optimization', 'evaluation'))

                # Write the string output to the file
                file.write(
                    '"""\n'
                    'Python script for the '
                    f'"{instance.configuration["label"]}" plan.\n\n'
                    'Generated from the pyanno4rt GUI.\n"""\n\n'
                    '# %% Internal package import\n\n'
                    'from pyanno4rt.base import TreatmentPlan\n'
                    'from pyanno4rt.gui import GraphicalUserInterface\n\n'
                    '# %% Initialization\n\n'
                    'tp = TreatmentPlan(\n\n'
                    f'    configuration={configuration},\n\n'
                    f'    optimization={optimization},\n\n'
                    f'    evaluation={evaluation}\n\n'
                    ')\n\n'
                    '# %% Workflow\n\n'
                    'tp.configure()\n'
                    'tp.model()\n'
                    'tp.optimize()\n'
                    'tp.evaluate()\n'
                    'tp.visualize()\n\n'
                    '# %% GUI\n\n'
                    'gui = GraphicalUserInterface()\n'
                    'gui.launch(tp)\n'
                    '')

                # Close the file stream
                file.close()

    def open_compare_window(self):
        """Open the plan comparison window."""

        # Get the baseline and reference plan for comparison
        baseline = self.plans[self.baseline_ledit.text()]
        reference = self.plans[self.reference_cbox.currentText()]

        # Check if the CT cubes have different shapes
        if (baseline.datahub.computed_tomography['cubeHU'].shape !=
                reference.datahub.computed_tomography['cubeHU'].shape):

            # Define the output string
            message = ("Baseline and reference plan have different CT cube "
                       "dimensions. Only plans with equal dimensions can be "
                       "compared!")

            # Show an information message box
            QMessageBox.information(self, 'pyanno4rt', message)

        # Check if any plan has not been (re-)optimized
        elif any(instance.datahub.state < 3 for instance in (
                baseline, reference)):

            # Define the output string
            message = ("Either baseline or reference plan have not been "
                       "(re-)optimized. Only plans with optimized 3D dose "
                       "distributions can be compared!")

            # Show an information message box
            QMessageBox.information(self, 'pyanno4rt', message)

        else:

            # Set the column titles
            self.compare_window.set_titles(
                self.baseline_ledit.text(), self.reference_cbox.currentText(),
                'Difference')

            # Add the treatment plans
            self.compare_window.add_plans(baseline, reference)

            # Set the position of the window
            self.compare_window.position()

            # Show the window
            self.compare_window.show()

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

    def configure_status_bar(self):
        """Configure the status bar."""

        # Reformat the status bar
        self.status_bar.reformat()

        # Initialize the loader label
        self.loader_label = QLabel()

        # Initialize and start the loader gif
        movie = QMovie(':/special_icons/icons_special/load.gif')
        movie.start()

        # Add the loader gif to the label
        self.loader_label.setMovie(movie)

        # Hide the label initially
        self.loader_label.hide()

        # Initialize the stop button
        self.stop_thread_pbutton = QPushButton()

        # Set the stop icon
        self.stop_thread_pbutton.setIcon(
            QIcon(':/special_icons/icons_special/stop.svg'))

        # Set the tooltip for the stop button
        self.stop_thread_pbutton.setToolTip("Stop the current process")

        # Set the cursor for the stop button
        self.stop_thread_pbutton.setCursor(QCursor(Qt.PointingHandCursor))

        # Initialize the logo label
        self.logo_label = QLabel()

        # Set the logo icon
        logo = QPixmap(':/special_icons/icons_special/logo_black_icon.png')

        # Rescale the logo icon
        logo = logo.scaled(int(logo.width()/4), int(logo.height()/4))

        # Add the logo to the label
        self.logo_label.setPixmap(logo)

        # Initialize the version label
        self.version_label = QLabel(f'"Amadeus" v{version("pyanno4rt")}')

        # Loop over the link push buttons
        for key, value in {
                'github_pbutton': (
                    ':/white_icons/icons_white/github.svg', 'Open Github'),
                'rtd_pbutton': (
                    ':/white_icons/icons_white/file-text.svg',
                    'Open Read the Docs'),
                'pypi_pbutton': (
                    ':/white_icons/icons_white/box.svg', 'Open PyPI')
                }.items():

            # Set the push button attribute
            setattr(self, key, QPushButton())

            # Get the push button object
            button = getattr(self, key)

            # Set the icon
            button.setIcon(QIcon(value[0]))

            # Set the tool tip
            button.setToolTip(value[1])

            # Set the cursor
            button.setCursor(QCursor(Qt.PointingHandCursor))

        # Set the stylesheets of the status bar elements
        self.set_styles({
            'loader_label': "QLabel {border: 0px;}",
            'stop_thread_pbutton': pbutton_statusbar,
            'logo_label': "QLabel {border: 0px;}",
            'version_label': "QLabel {border: 0px; font-size: 10pt;}",
            'github_pbutton': pbutton_statusbar,
            'rtd_pbutton': pbutton_statusbar,
            'pypi_pbutton': pbutton_statusbar})

        # Loop over the status bar elements
        for element in (
                self.loader_label, self.stop_thread_pbutton, SpacerFrame(),
                self.logo_label, self.version_label, SpacerFrame(),
                self.github_pbutton, self.rtd_pbutton, self.pypi_pbutton,
                SpacerFrame()):

            # Add the element to the status bar
            self.status_bar.addPermanentWidget(element)

        # Set the initial status bar message
        self.status_bar.showMessage(
            "Ready to load/select/create a treatment plan ...")

    def stop_thread_by_user(self):
        """Stop the thread by user interaction."""

        # Turn the stop button and loader label off
        self.run_stop_and_loader(False)

        # Needs to be implemented still
        # ...

    def open_github_link(self):
        """Open the pyanno4rt Github page."""

        # Open the link with the default browser
        webopen('https://github.com/pyanno4rt/pyanno4rt')

    def open_rtd_link(self):
        """Open the pyanno4rt Read the Docs page."""

        # Open the link with the default browser
        webopen('https://pyanno4rt.readthedocs.io/en/latest/')

    def open_pypi_link(self):
        """Open the pyanno4rt PyPI page."""

        # Open the link with the default browser
        webopen('https://pypi.org/project/pyanno4rt/')

    def position(self):
        """Set the window position."""

        # Get the frame geometry
        geometry = self.frameGeometry()

        # Get the screen number from the cursor position
        screen = QApplication.desktop().screenNumber(
            QApplication.desktop().cursor().pos())

        # Move the geometry center according to the application window
        geometry.moveCenter(
            QApplication.desktop().screenGeometry(screen).center())

        # Move the window to the top left of the geometry
        self.move(geometry.topLeft())


class SpacerFrame(QFrame):
    """Custom QFrame class for the spacing of the status bar fields."""

    def __init__(self):

        # Run the constructor from the superclass
        super().__init__()

        # Set the stylesheet
        self.setStyleSheet("QFrame {border: 0px solid;}")

        # Set the frame shape
        self.setFrameShape(self.VLine | self.Sunken)


class ConsoleWindowLogHandler(Handler, QObject):
    """
    Handler class for displaying the logging messages in the status bar.

    Attributes
    ----------
    stream : object of class :class:`~PyQt5.QtCore.pyqtSignal`
        The object used to represent the output stream signal.
    """

    # Initialize the 'stream' signal
    stream = pyqtSignal(str)

    def __init__(self):

        # Run the constructor from the 'Handler' superclass
        Handler.__init__(self)

        # Run the constructor of the 'QObject' superclass
        QObject.__init__(self)

    def emit(
            self,
            record):
        """
        Emit the log output message.

        Parameters
        ----------
        record : object of class :class:`~logging.LogRecord`
            The object used to represent the log record.
        """

        # Emit the message from the 'stream' signal
        self.stream.emit(str(record.getMessage()))


class Worker(QThread):
    """
    Worker thread class for running tasks.

    Parameters
    ----------
    task : object of class :class:`~method`
        The task to be performed in the worker thread.
    """

    # Initialize the 'failed' signal
    failed = pyqtSignal(Exception)

    # Initialize the 'finished' signal
    finished = pyqtSignal()

    def __init__(
            self,
            task):

        # Run the constructor from the superclass
        super().__init__()

        # Initialize the worker task
        self.task = task

    def run(self):
        """Run the worker thread."""

        try:

            # Run the task
            self.task()

            # Emit the 'finished' signal
            self.finished.emit()

        except Exception as exception:

            # Emit the 'failed' signal
            self.failed.emit(exception)
