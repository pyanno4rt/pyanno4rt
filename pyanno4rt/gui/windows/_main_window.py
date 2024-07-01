"""Main window."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from functools import partial
from os.path import abspath, dirname
from json import loads
from numpy import zeros
from PyQt5.QtCore import QDir, QEvent, Qt
from PyQt5.QtGui import QCursor, QIcon, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QHeaderView, QListWidget,
    QListWidgetItem, QMainWindow, QMenu, QMessageBox, QSpinBox)

# %% Internal package import

from pyanno4rt.base import TreatmentPlan
from pyanno4rt.gui.compilations.main_window import Ui_main_window
from pyanno4rt.gui.custom_widgets import DVHWidget, SliceWidget
from pyanno4rt.gui.styles._custom_styles import (
    cbox, ledit, pbutton_menu, pbutton_composer, pbutton_workflow, sbox,
    selector, tab, tbutton_composer, tbutton_workflow)
from pyanno4rt.gui.windows import (
    CompareWindow, InfoWindow, LogWindow, PlanCreationWindow, SettingsWindow,
    TreeWindow)
from pyanno4rt.gui.windows.components import component_window_map
from pyanno4rt.optimization.components import component_map
from pyanno4rt.tools import (
    add_square_brackets, apply, copycat, load_list_from_file, snapshot)

# %% Class definition


class MainWindow(QMainWindow, Ui_main_window):
    """
    Main window for the GUI.

    This class creates the main window for the graphical user interface, \
    including logo, labels, and input/control elements.

    Parameters
    ----------
    treatment_plan : object of class `TreatmentPlan`
        Instance of the class `TreatmentPlan`, which provides methods and \
        classes to generate treatment plans.

    application : object of class `SpyderQApplication`
        Instance of the class `SpyderQApplication` for managing control flow \
        and main settings of the graphical user interface.
    """

    def __init__(
            self,
            treatment_plan,
            application=None):

        # Run the constructor from the superclass
        super().__init__()

        # Get the application from the argument
        self.application = application

        # Build the UI main window
        self.setupUi(self)

        # Set the window icon
        self.setWindowIcon(QIcon('./logo/logo_white_icon.png'))

        # Initialize the GUI plans dictionary
        self.plans = {}

        # Initialize the optimized plans list
        self.optimized_plans = []

        # Initialize the GUI optimization components
        self.plan_components = {}

        # Initialize the current component window
        self.current_component_window = None

        # Initialize the last selected plan
        self.last_selection = ''

        # Initialize the child windows
        self.plan_creation_window = PlanCreationWindow(self)
        self.settings_window = SettingsWindow(self)
        self.info_window = InfoWindow(self)
        self.compare_window = CompareWindow(self)
        self.parameter_window = TreeWindow('Plan Parameters Viewer', self)
        self.plan_window = TreeWindow('Plan Data Viewer', self)
        self.model_data_window = TreeWindow('Model Data Viewer', self)
        self.feature_map_window = TreeWindow('Feature Maps Viewer', self)
        self.log_window = LogWindow(self)

        # Initialize the custom widgets
        self.dvh_widget = DVHWidget(self)
        self.slice_widget = SliceWidget(self)

        # Insert the custom widgets into the viewer
        self.tab_dvh_layout.insertWidget(0, self.dvh_widget)
        self.tab_slices_layout.insertWidget(0, self.slice_widget)

        # Get the base dictionaries
        self.base_configuration = self.transform_configuration_to_dict()
        self.base_optimization = self.transform_optimization_to_dict()
        self.base_evaluation = self.transform_evaluation_to_dict()

        # Loop over the tab widgets
        for widget in ('composer_widget', 'tab_workflow', 'viewer_widget'):

            # Set the initial tab index
            getattr(self, widget).setCurrentIndex(0)

        # Loop over the QComboBox and QSpinBox elements
        for box in ('log_level_cbox', 'modality_cbox', 'nfx_sbox',
                    'method_cbox', 'solver_cbox', 'algorithm_cbox',
                    'init_strat_cbox', 'ref_plan_cbox', 'max_iter_sbox',
                    'dvh_type_cbox', 'n_points_sbox', 'opacity_sbox'):

            # Install the custom event filters
            getattr(self, box).installEventFilter(self)

        # Disable some fields initially
        self.set_disabled((
            'save_pbutton', 'drop_pbutton', 'update_configuration_pbutton',
            'update_optimization_pbutton', 'update_evaluation_pbutton',
            'reset_configuration_pbutton', 'reset_optimization_pbutton',
            'reset_evaluation_pbutton', 'initialize_pbutton',
            'configure_pbutton', 'optimize_pbutton', 'evaluate_pbutton',
            'visualize_pbutton', 'actions_show_parameter_tbutton',
            'actions_show_plan_tbutton', 'actions_show_fmap_tbutton',
            'actions_show_model_data_tbutton', 'actions_show_log_tbutton',
            'comp_show_parameter_tbutton', 'comp_show_plan_tbutton',
            'comp_show_fmap_tbutton', 'comp_show_model_data_tbutton',
            'comp_show_log_tbutton', 'compare_pbutton', 'baseline_cbox',
            'reference_cbox', 'components_minus_tbutton',
            'components_edit_tbutton', 'init_fluence_ledit',
            'init_fluence_tbutton', 'ref_plan_cbox', 'opacity_sbox',
            'slice_selection_sbar'))

        # Disable the tab widgets initially
        self.composer_widget.widget(0).setEnabled(False)
        self.composer_widget.widget(1).setEnabled(False)
        self.composer_widget.widget(2).setEnabled(False)

        # Set the line edit cursor positions to zero
        self.set_zero_line_cursor((
            'plan_ledit', 'img_path_ledit', 'dose_path_ledit',
            'init_fluence_ledit', 'lower_var_ledit', 'upper_var_ledit',
            'ref_vol_ledit', 'ref_dose_ledit', 'display_segments_ledit'))

        # Set the stylesheets
        self.set_styles({'composer_widget': tab,
                         'tab_workflow': tab,
                         'viewer_widget': tab,
                         'plan_ledit': ledit,
                         'log_level_cbox': cbox,
                         'modality_cbox': cbox,
                         'nfx_sbox': sbox,
                         'img_path_ledit': ledit,
                         'img_res_ledit': ledit,
                         'dose_path_ledit': ledit,
                         'dose_res_ledit': ledit,
                         'method_cbox': cbox,
                         'solver_cbox': cbox,
                         'algorithm_cbox': cbox,
                         'init_strat_cbox': cbox,
                         'init_fluence_ledit': ledit,
                         'ref_plan_cbox': cbox,
                         'lower_var_ledit': ledit,
                         'lower_var_tbutton': tbutton_composer,
                         'upper_var_ledit': ledit,
                         'upper_var_tbutton': tbutton_composer,
                         'max_iter_sbox': sbox,
                         'tolerance_ledit': ledit,
                         'dvh_type_cbox': cbox,
                         'n_points_sbox': sbox,
                         'ref_vol_ledit': ledit,
                         'ref_dose_ledit': ledit,
                         'display_segments_ledit': ledit,
                         'opacity_sbox': sbox,
                         'load_pbutton': pbutton_menu,
                         'save_pbutton': pbutton_menu,
                         'drop_pbutton': pbutton_menu,
                         'plan_select_cbox': selector,
                         'settings_pbutton': pbutton_menu,
                         'info_pbutton': pbutton_menu,
                         'exit_pbutton': pbutton_menu,
                         'img_path_tbutton': tbutton_composer,
                         'dose_path_tbutton': tbutton_composer,
                         'components_plus_tbutton': tbutton_composer,
                         'components_minus_tbutton': tbutton_composer,
                         'components_edit_tbutton': tbutton_composer,
                         'init_fluence_tbutton': tbutton_composer,
                         'lower_var_tbutton': tbutton_composer,
                         'upper_var_tbutton': tbutton_composer,
                         'update_configuration_pbutton': pbutton_composer,
                         'update_optimization_pbutton': pbutton_composer,
                         'update_evaluation_pbutton': pbutton_composer,
                         'reset_configuration_pbutton': pbutton_composer,
                         'reset_optimization_pbutton': pbutton_composer,
                         'reset_evaluation_pbutton': pbutton_composer,
                         'clear_configuration_pbutton': pbutton_composer,
                         'clear_optimization_pbutton': pbutton_composer,
                         'clear_evaluation_pbutton': pbutton_composer,
                         'initialize_pbutton': pbutton_workflow,
                         'configure_pbutton': pbutton_workflow,
                         'optimize_pbutton': pbutton_workflow,
                         'evaluate_pbutton': pbutton_workflow,
                         'visualize_pbutton': pbutton_workflow,
                         'actions_show_parameter_tbutton': tbutton_workflow,
                         'actions_show_plan_tbutton': tbutton_workflow,
                         'actions_show_model_data_tbutton': tbutton_workflow,
                         'actions_show_fmap_tbutton': tbutton_workflow,
                         'actions_show_log_tbutton': tbutton_workflow,
                         'comp_show_parameter_tbutton': tbutton_workflow,
                         'comp_show_plan_tbutton': tbutton_workflow,
                         'comp_show_model_data_tbutton': tbutton_workflow,
                         'comp_show_fmap_tbutton': tbutton_workflow,
                         'comp_show_log_tbutton': tbutton_workflow,
                         'baseline_cbox': cbox,
                         'reference_cbox': cbox,
                         'compare_pbutton': pbutton_workflow,
                         'expand_tree_pbutton': pbutton_composer,
                         'collapse_tree_pbutton': pbutton_composer,
                         'close_tree_pbutton': pbutton_composer,
                         'close_log_pbutton': pbutton_composer,
                         'reset_settings_pbutton': pbutton_composer,
                         'save_settings_pbutton': pbutton_composer,
                         'close_text_pbutton': pbutton_composer,
                         'close_info_pbutton': pbutton_composer,
                         'create_plan_pbutton': pbutton_composer,
                         'close_plan_pbutton': pbutton_composer,
                         'new_plan_ref_cbox': cbox,
                         'new_modality_cbox': cbox,
                         'new_img_path_tbutton': tbutton_composer,
                         'new_dose_path_tbutton': tbutton_composer})

        # Add dropdown menu to component adding button
        menu = QMenu()
        for key in component_map:
            menu.addAction(key, partial(self.open_component_window, key))
        self.components_plus_tbutton.setPopupMode(2)
        self.components_plus_tbutton.setMenu(menu)

        # Connect the event signals
        self.connect_signals()

        # Check if an initial treatment plan has been specified
        if treatment_plan:

            # Set the initial treatment plan
            self.set_initial_plan(treatment_plan)

        # Set the initial window size
        # self.resize(1440, 900)

        # Show the GUI
        self.showMaximized()

    def connect_signals(self):
        """Connect the event signals to the GUI elements."""

        # Loop over the fieldnames with 'clicked' events
        for key, value in {
                'load_pbutton': self.load_tpi,
                'save_pbutton': self.save_tpi,
                'settings_pbutton': self.open_settings_window,
                'info_pbutton': self.open_info_window,
                'compare_pbutton': self.open_compare_window,
                'exit_pbutton': self.exit_window,
                'drop_pbutton': self.open_question_dialog,
                'update_configuration_pbutton': self.open_question_dialog,
                'update_optimization_pbutton': self.open_question_dialog,
                'update_evaluation_pbutton': self.open_question_dialog,
                'reset_configuration_pbutton': self.open_question_dialog,
                'reset_optimization_pbutton': self.open_question_dialog,
                'reset_evaluation_pbutton': self.open_question_dialog,
                'clear_configuration_pbutton': self.open_question_dialog,
                'clear_optimization_pbutton': self.open_question_dialog,
                'clear_evaluation_pbutton': self.open_question_dialog,
                'img_path_tbutton': self.add_imaging_path,
                'dose_path_tbutton': self.add_dose_matrix_path,
                'components_minus_tbutton': self.remove_component,
                'components_edit_tbutton': self.edit_component,
                'init_fluence_tbutton': self.add_initial_fluence_vector,
                'lower_var_tbutton': self.add_lower_var_bounds,
                'upper_var_tbutton': self.add_upper_var_bounds,
                'initialize_pbutton': self.initialize,
                'configure_pbutton': self.configure,
                'optimize_pbutton': self.optimize,
                'evaluate_pbutton': self.evaluate,
                'visualize_pbutton': self.visualize,
                'actions_show_parameter_tbutton': self.open_parameter_window,
                'actions_show_plan_tbutton': self.open_plan_window,
                'actions_show_model_data_tbutton': self.open_model_data_window,
                'actions_show_fmap_tbutton': self.open_feature_map_window,
                'actions_show_log_tbutton': self.open_log_window,
                'comp_show_parameter_tbutton': self.open_parameter_window,
                'comp_show_plan_tbutton': self.open_plan_window,
                'comp_show_model_data_tbutton': self.open_model_data_window,
                'comp_show_fmap_tbutton': self.open_feature_map_window,
                'comp_show_log_tbutton': self.open_log_window
                }.items():

            # Connect the 'clicked' signal
            getattr(self, key).clicked.connect(value)

        # Loop over the fieldnames with 'currentIndexChanged' events
        for key, value in {
                'plan_select_cbox': self.update_reference_plans,
                'method_cbox': self.update_by_method,
                'solver_cbox': self.update_by_solver
                }.items():

            # Connect the 'currentIndexChanged' signal
            getattr(self, key).currentIndexChanged.connect(value)

        # Loop over the fieldnames with 'currentTextChanged' events
        for key, value in {
                'plan_select_cbox': self.select_plan,
                'init_strat_cbox': self.update_by_initial_strategy,
                'ref_plan_cbox': self.update_by_reference,
                'baseline_cbox': self.update_compare_button,
                'reference_cbox': self.update_compare_button
                }.items():

            # Connect the 'currentTextChanged' signal
            getattr(self, key).currentTextChanged.connect(value)

        # Loop over the fieldnames with 'textChanged' events
        for key, value in {
                'plan_ledit': self.update_init_button,
                'img_path_ledit': self.update_init_button,
                'dose_path_ledit': self.update_init_button,
                'dose_res_ledit': self.update_init_button,
                'init_fluence_ledit': self.update_by_initial_fluence
                }.items():

            # Connect the 'textChanged' signal
            getattr(self, key).textChanged.connect(value)

        # Loop over the fieldnames with 'valueChanged' events
        for key, value in {
                'opacity_sbox': self.slice_widget.change_dose_opacity,
                'slice_selection_sbar': self.slice_widget.change_image_slice
                }.items():

            # Connect the 'valueChanged' events
            getattr(self, key).valueChanged.connect(value)

        # Loop over the fieldnames with 'currentItemChanged' events
        for key, value in {
                'components_lwidget': (lambda: self.set_enabled((
                    'components_minus_tbutton', 'components_edit_tbutton')))
                }.items():

            # Connect the 'currentItemChanged' events
            getattr(self, key).currentItemChanged.connect(value)

    def eventFilter(
            self,
            source,
            event):
        """
        Customize the event filters.

        Parameters
        ----------
        source : ...
            ...

        event : ...
            ...

        Returns
        -------
        ...
        """

        # Check if a mouse wheel event applies to QComboBox or QSpinBox
        if (event.type() == QEvent.Wheel and
                isinstance(source, (QComboBox, QSpinBox))):

            # Filter the event
            return True

        # Else, return the even
        return super().eventFilter(source, event)

    def mousePressEvent(self, event):

        super(QListWidget, self.components_lwidget).mousePressEvent(event)
        super(QListWidget, self.display_metrics_lwidget).mousePressEvent(event)

        if not self.components_lwidget.indexAt(event.pos()).isValid():
            self.components_lwidget.clearSelection()

        if not self.display_metrics_lwidget.indexAt(event.pos()).isValid():
            self.display_metrics_lwidget.clearSelection()

    def set_initial_plan(
            self,
            treatment_plan):
        """
        Set the initial treatment plan in the GUI.

        Parameters
        ----------
        treatment_plan : ...
            ...
        """

        # Check if the treatment plan is a single instance
        if isinstance(treatment_plan, TreatmentPlan):

            # Activate the treatment plan in the GUI
            self.activate(treatment_plan)

        # Else, check if the treatment plan is a list of instances
        elif (isinstance(treatment_plan, list) and
              all(isinstance(plan, TreatmentPlan) for plan in treatment_plan)):

            # Activate each treatment plan in the GUI
            apply(self.activate, treatment_plan)

    def set_enabled(
            self,
            fieldnames):
        """
        Enable multiple fields by their names.

        Parameters
        ----------
        fieldnames : ...
            ...
        """

        # Loop over the passed fieldnames
        for name in fieldnames:

            # Get the attribute and set it enabled
            getattr(self, name).setEnabled(True)

    def set_disabled(
            self,
            fieldnames):
        """
        Disable multiple fields by their names.

        Parameters
        ----------
        fieldnames : ...
            ...
        """

        # Loop over the passed fieldnames
        for name in fieldnames:

            # Get the attribute and set it disabled
            getattr(self, name).setEnabled(False)

    def set_zero_line_cursor(
            self,
            fieldnames):
        """
        Set the line edit cursor positions to zero.

        Parameters
        ----------
        fieldnames : ...
            ...
        """

        # Loop over the passed fieldnames
        for name in fieldnames:

            # Get the attribute and set the cursor position to zero
            getattr(self, name).setCursorPosition(0)

    def set_styles(
            self,
            key_value_pairs):
        """
        Set the element stylesheets from key-value pairs.

        Parameters
        ----------
        key_value_pairs : ...
            ...
        """

        # Loop over the dictionary items
        for key, value in key_value_pairs.items():

            # Check if the key does not refer to a tab widget
            if key not in ('composer_widget', 'tab_workflow',
                           'viewer_widget', 'expand_tree_pbutton',
                           'collapse_tree_pbutton', 'close_tree_pbutton',
                           'close_log_pbutton', 'reset_settings_pbutton',
                           'save_settings_pbutton', 'close_text_pbutton',
                           'close_info_pbutton', 'create_plan_pbutton',
                           'close_plan_pbutton', 'new_plan_ref_cbox',
                           'new_modality_cbox', 'new_img_path_tbutton',
                           'new_dose_path_tbutton'):

                # Get the attribute and set the stylesheet
                getattr(self, key).setStyleSheet(value)

            elif key in ('composer_widget', 'tab_workflow', 'viewer_widget'):

                # Get the tab bar of the attribute and set the stylesheet
                getattr(self, key).tabBar().setStyleSheet(value)

            elif key in ('expand_tree_pbutton', 'collapse_tree_pbutton',
                         'close_tree_pbutton'):

                for window in (
                        self.parameter_window, self.plan_window,
                        self.model_data_window, self.feature_map_window):

                    # Get the attribute and set the stylesheet
                    getattr(window, key).setStyleSheet(value)

            elif key == 'close_log_pbutton':

                # Get the attribute and set the stylesheet
                getattr(self.log_window, key).setStyleSheet(value)

            elif key in ('reset_settings_pbutton', 'save_settings_pbutton'):

                # Get the attribute and set the stylesheet
                getattr(self.settings_window, key).setStyleSheet(value)

            elif key == 'close_text_pbutton':

                for window in (
                        self.parameter_window, self.plan_window,
                        self.model_data_window, self.feature_map_window):

                    # Get the attribute and set the stylesheet
                    getattr(window.text_window, key).setStyleSheet(value)

            elif key == 'close_info_pbutton':

                # Get the attribute and set the stylesheet
                getattr(self.info_window, key).setStyleSheet(value)

            elif key in ('create_plan_pbutton', 'close_plan_pbutton',
                         'new_plan_ref_cbox', 'new_modality_cbox',
                         'new_img_path_tbutton', 'new_dose_path_tbutton'):

                # Get the attribute and set the stylesheet
                getattr(self.plan_creation_window, key).setStyleSheet(value)

    def activate(
            self,
            treatment_plan):
        """
        Activate a treatment plan instance in the GUI.

        Parameters
        ----------
        treatment_plan : ...
            ...
        """

        # Get the plan label
        label = treatment_plan.configuration['label']

        # Check if the instance is not yet included in the selector
        if label not in (self.plan_select_cbox.itemText(i)
                         for i in range(self.plan_select_cbox.count())):

            # Add the instance to the dictionaries
            self.plans[label] = treatment_plan
            self.plan_components[label] = {}

            # Add the loaded instance label to the selector
            self.plan_select_cbox.insertItem(0, label)

            # 
            self.plan_select_cbox.removeItem(self.plan_select_cbox.count()-1)

            # Select the loaded instance label
            self.plan_select_cbox.setCurrentText(label)

            # Sort the items in the selector alphabetically
            self.plan_select_cbox.model().sort(0)

            # Set the icon path on the red frame
            icon_path = (":/lightred_icons/icons_lightred/plus-square.svg")

            # Initialize the icon object
            icon = QIcon()

            # Add the pixmap to the icon
            icon.addPixmap(QPixmap(icon_path), QIcon.Normal, QIcon.Off)

            # 
            self.plan_select_cbox.insertItem(
                self.plan_select_cbox.count(), icon, 'Create new plan')

        else:

            # Select the loaded instance label
            self.plan_select_cbox.setCurrentText(label)

    def load_tpi(self):
        """Load the treatment plan from a snapshot folder."""

        # Get the loading folder path
        path = QFileDialog.getExistingDirectory(
            self, 'Select a directory for loading')

        # Check if the folder name exists
        if path:

            # Copycat the treatment plan
            tp_copy = copycat(TreatmentPlan, path)

            # Activate the copycat treatment plan
            self.activate(tp_copy)

    def save_tpi(self):
        """Save the treatment plan to a snapshot folder."""

        # Get the saving folder path
        path = QFileDialog.getExistingDirectory(
            self, 'Select a directory for saving')

        # Check if the folder name exists
        if path:

            # Get the additional save arguments from the settings
            includes = self.settings_window.current[3]

            # Make a snapshot of the treatment plan instance
            snapshot(self.plans[self.plan_ledit.text()],
                     ''.join((path, '/')), *includes)

    def drop_tpi(self):
        """Remove the current treatment plan."""

        # Get the instance label
        label = self.plan_ledit.text()

        # Check if the label is included in the plan dictionary
        if label in (*self.plans,):

            # Delete the instance from the plan dictionary
            del self.plans[label]

            # Reset the selector index to the default
            self.plan_select_cbox.setCurrentIndex(-1)

            # 
            for i in range(self.plan_select_cbox.count()):

                if label == self.plan_select_cbox.itemText(i):

                    # Remove the instance item from the selector
                    self.plan_select_cbox.removeItem(i)

            # 
            if self.plan_select_cbox.itemText(0) == 'Create new plan':

                # 
                self.plan_select_cbox.insertItem(0, '')

                # Reset the selector index to the default
                self.plan_select_cbox.setCurrentIndex(0)

            # Check if the label is included in the optimized plans list
            if label in self.optimized_plans:

                # Remove the label from the list
                self.optimized_plans.remove(label)

                # Remove the label from the comparison plans
                self.update_comparison_plans()

    def select_plan(self):
        """Select a treatment plan."""

        # Get the selected plan
        selection = self.plan_select_cbox.currentText()

        # 
        if selection == 'Create new plan':

            # 
            if self.last_selection == '':

                # 
                self.plan_select_cbox.setCurrentIndex(-1)

            else:

                # 
                self.plan_select_cbox.setCurrentText(self.last_selection)

            # 
            self.open_plan_creation_window()

        # Check if the selector is not set to default
        elif selection != '':

            # 
            for i in range(self.plan_select_cbox.count()):

                # 
                if self.plan_select_cbox.itemText(i) == '':

                    # 
                    self.plan_select_cbox.removeItem(i)

            # 
            self.last_selection = selection

            # Change the treatment plan label to the instance label
            self.plan_ledit.setText(selection)

            # 
            self.plan_ledit.setReadOnly(True)

            # Get the treatment plan instance
            instance = self.plans[selection]

            # Set the configuration, optimization and evaluation tab fields
            self.set_configuration()
            self.set_optimization()
            self.set_evaluation()

            # Reset the slice widget
            self.slice_widget.reset_images()

            # Reset the DVH widget
            self.dvh_widget.reset_dvh()

            # Enable specific fields
            self.set_enabled((
                'save_pbutton', 'drop_pbutton', 'configure_pbutton',
                'update_configuration_pbutton', 'update_optimization_pbutton',
                'update_evaluation_pbutton', 'reset_configuration_pbutton',
                'reset_optimization_pbutton', 'reset_evaluation_pbutton',
                'actions_show_parameter_tbutton',
                'comp_show_parameter_tbutton'))

            # Disable specific fields
            self.set_disabled((
                'initialize_pbutton', 'optimize_pbutton', 'evaluate_pbutton',
                'visualize_pbutton', 'actions_show_plan_tbutton',
                'actions_show_log_tbutton', 'actions_show_model_data_tbutton',
                'actions_show_fmap_tbutton', 'comp_show_plan_tbutton',
                'comp_show_log_tbutton', 'comp_show_model_data_tbutton',
                'comp_show_fmap_tbutton'))

            # Disable the tab widgets
            self.composer_widget.widget(0).setEnabled(True)
            self.composer_widget.widget(1).setEnabled(True)
            self.composer_widget.widget(2).setEnabled(True)

            # Set the line edit cursor positions to zero
            self.set_zero_line_cursor((
                'plan_ledit', 'img_path_ledit', 'dose_path_ledit',
                'init_fluence_ledit', 'lower_var_ledit', 'upper_var_ledit',
                'ref_vol_ledit', 'ref_dose_ledit', 'display_segments_ledit'))

            # Check if the selected plan has been initialized
            if instance.datahub and instance.datahub.logger:

                # Update the log output
                self.log_window.update_log_output()

                # Enable the plan data and logging tool buttons
                self.set_enabled((
                    'actions_show_plan_tbutton', 'actions_show_log_tbutton',
                    'comp_show_plan_tbutton', 'comp_show_log_tbutton'))

            # Check if the selected plan has been configured
            if (instance.datahub and all(isinstance(getattr(
                    instance.datahub, unit), dict)
                    for unit in ('computed_tomography', 'segmentation'))):

                # Get the CT dictionary
                computed_tomography = instance.datahub.computed_tomography

                # Get the axial dimension of the CT cube
                axial_length = computed_tomography['cube_dimensions'][2]

                # Get the segmentation dictionary
                segmentation = instance.datahub.segmentation

                # Add the CT cube to the slice widget
                self.slice_widget.add_ct(computed_tomography['cube'])

                # Add the segments to the slice widget
                self.slice_widget.add_segments(
                    computed_tomography, segmentation)

                # Set the range of the slice selection scrollbar
                self.slice_selection_sbar.setRange(0, axial_length-1)

                # Set the initial scrollbar value
                self.slice_selection_sbar.setValue(int((axial_length-1)/2))

                # Set the initial position label
                self.slice_selection_pos.setText(''.join((
                    'z = ', str(self.slice_widget.slice), ' mm')))

                # Enable the 'optimize' button
                self.optimize_pbutton.setEnabled(True)

                # Check if the selected plan has been optimized
                if isinstance(instance.datahub.optimization, dict):

                    # Get the optimized dose array
                    optimized_dose = instance.datahub.optimization[
                        'optimized_dose']

                    # Check if the plan has already been optimized
                    if selection not in self.optimized_plans:

                        # Append the treatment plan to the optimized plans list
                        self.optimized_plans.append(selection)

                        # Update the comparison plans
                        self.update_comparison_plans()

                    # Add the dose image to the slice widget
                    self.slice_widget.add_dose(optimized_dose)

                    # Enable the 'evaluate' and 'visualize' buttons
                    self.set_enabled(('evaluate_pbutton', 'visualize_pbutton'))

                # Update the images of the slice widget
                self.slice_widget.update_images()

                # Check if the selected plan has been evaluated
                if isinstance(instance.datahub.dose_histogram, dict):

                    # Add the style and input data to the DVH widget
                    self.dvh_widget.add_style_and_data(
                        instance.datahub.dose_histogram)

                    # Update the plot of the DVH widget
                    self.dvh_widget.update_dvh()

            # Check if the selected plan includes data models
            if instance.datahub and any(unit is not None for unit in (
                    instance.datahub.datasets, instance.datahub.feature_maps,
                    instance.datahub.model_instances,
                    instance.datahub.model_inspections,
                    instance.datahub.model_evaluations)):

                # Enable the model data and feature maps tool button
                self.set_enabled((
                    'actions_show_model_data_tbutton',
                    'actions_show_fmap_tbutton',
                    'comp_show_model_data_tbutton',
                    'comp_show_fmap_tbutton'))

        else:

            # 
            self.last_selection = ''

            # 
            self.plan_ledit.setReadOnly(False)

            # Clear the configuration, optimization and evaluation tab fields
            self.clear_configuration()
            self.clear_optimization()
            self.clear_evaluation()

            # Disable specific fields
            self.set_disabled((
                'save_pbutton', 'drop_pbutton', 'update_configuration_pbutton',
                'update_optimization_pbutton', 'update_evaluation_pbutton',
                'reset_configuration_pbutton', 'reset_optimization_pbutton',
                'reset_evaluation_pbutton', 'initialize_pbutton',
                'configure_pbutton', 'optimize_pbutton', 'evaluate_pbutton',
                'visualize_pbutton', 'actions_show_parameter_tbutton',
                'actions_show_plan_tbutton', 'actions_show_fmap_tbutton',
                'actions_show_model_data_tbutton', 'actions_show_log_tbutton',
                'comp_show_parameter_tbutton', 'comp_show_plan_tbutton',
                'comp_show_fmap_tbutton', 'comp_show_model_data_tbutton',
                'comp_show_log_tbutton', 'init_fluence_ledit',
                'init_fluence_tbutton', 'ref_plan_cbox'))

            # Disable the tab widgets
            self.composer_widget.widget(0).setEnabled(False)
            self.composer_widget.widget(1).setEnabled(False)
            self.composer_widget.widget(2).setEnabled(False)

            # Reset the slice widget
            self.slice_widget.reset_images()

            # Reset the DVH widget
            self.dvh_widget.reset_dvh()

    def initialize(self):
        """
        Initialize the treatment plan.

        Returns
        -------
        bool
            Indicator for the success of the initialization.
        """

        # Set the waiting cursor
        QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))

        try:

            # Initialize the treatment plan
            treatment_plan = TreatmentPlan(
                self.transform_configuration_to_dict(),
                self.transform_optimization_to_dict(),
                self.transform_evaluation_to_dict())

        except Exception as error:

            # Set back to the arrow cursor
            QApplication.restoreOverrideCursor()

            # Show a warning message box
            QMessageBox.warning(self, "pyanno4rt", str(error))

            return False

        # Activate the treatment plan
        self.activate(treatment_plan)

        # Set back to the arrow cursor
        QApplication.restoreOverrideCursor()

        return True

    def configure(self):
        """
        Configure the treatment plan.

        Returns
        -------
        bool
            Indicator for the success of the configuration.
        """

        # Set the waiting cursor
        QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))

        try:

            # Get the treatment plan instance
            instance = self.plans[self.plan_ledit.text()]

            # Configure the treatment plan
            instance.configure()

            # Get the CT dictionary
            computed_tomography = instance.datahub.computed_tomography

            # Get the axial dimension of the CT cube
            axial_length = computed_tomography['cube_dimensions'][2]

            # Get the segmentation dictionary
            segmentation = instance.datahub.segmentation

            # Reset the slice widget
            self.slice_widget.reset_images()

            # Add the CT cube to the slice widget
            self.slice_widget.add_ct(computed_tomography['cube'])

            # Add the segments to the slice widget
            self.slice_widget.add_segments(computed_tomography, segmentation)

            # Set the range of the slice selection scrollbar
            self.slice_selection_sbar.setRange(0, axial_length-1)

            # Set the initial scrollbar value
            self.slice_selection_sbar.setValue(int((axial_length-1)/2))

            # Set the initial position label
            self.slice_selection_pos.setText(''.join((
                str(computed_tomography['z'][int((axial_length-1)/2)]),
                ' mm')))

        except Exception as error:

            # Set back to the arrow cursor
            QApplication.restoreOverrideCursor()

            # Show a warning message box
            QMessageBox.warning(self, "pyanno4rt", str(error))

            return False

        # Update the images of the slice widget
        self.slice_widget.update_images()

        # Update the output of the log window
        self.log_window.update_log_output()

        # Enable specific fields
        self.set_enabled((
            'optimize_pbutton', 'actions_show_plan_tbutton',
            'actions_show_log_tbutton', 'comp_show_plan_tbutton',
            'comp_show_log_tbutton'))

        # Set the line edit cursor positions to zero
        self.set_zero_line_cursor((
            'plan_ledit', 'img_path_ledit', 'dose_path_ledit',
            'init_fluence_ledit', 'lower_var_ledit', 'upper_var_ledit',
            'ref_vol_ledit', 'ref_dose_ledit', 'display_segments_ledit'))

        # Set back to the arrow cursor
        QApplication.restoreOverrideCursor()

        return True

    def optimize(self):
        """
        Optimize the treatment plan.

        Returns
        -------
        bool
            Indicator for the success of the optimization.
        """

        # Set the waiting cursor
        QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))

        try:

            # Get the treatment plan instance
            instance = self.plans[self.plan_ledit.text()]

            # Optimize the treatment plan
            instance.optimize()

            # Check if the plan has already been optimized
            if self.plan_ledit.text() not in self.optimized_plans:

                # Append the treatment plan to the optimized plans list
                self.optimized_plans.append(self.plan_ledit.text())

                # Update the comparison plans
                self.update_comparison_plans()

            # Get the optimized dose array
            optimized_dose = instance.datahub.optimization['optimized_dose']

            # Check if any dose contours have been computed
            if self.slice_widget.dose_contours:

                # Loop over the dose contours
                for contour in self.slice_widget.dose_contours:

                    # Update the dose contour lines
                    contour.setData(zeros(self.slice_widget.dose_cube[
                        :, :, self.slice_widget.slice].shape))

            self.slice_widget.dose_cube = None
            self.slice_widget.dose_contours = None

            # Add the dose image to the slice widget
            self.slice_widget.add_dose(optimized_dose)

        except Exception as error:

            # Set back to the arrow cursor
            QApplication.restoreOverrideCursor()

            # Show a warning message box
            QMessageBox.warning(self, "pyanno4rt", str(error))

            return False

        # Update the images of the slice widget
        self.slice_widget.update_images()

        # Update the output of the log window
        self.log_window.update_log_output()

        # Enable the evaluation and visualization button
        self.set_enabled(('evaluate_pbutton', 'visualize_pbutton'))

        # Check if the selected plan includes data models
        if instance.datahub and any(unit is not None for unit in (
                instance.datahub.datasets, instance.datahub.feature_maps,
                instance.datahub.model_instances,
                instance.datahub.model_inspections,
                instance.datahub.model_evaluations)):

            # Enable the model data and feature maps tool button
            self.set_enabled((
                'actions_show_model_data_tbutton',
                'actions_show_fmap_tbutton',
                'comp_show_model_data_tbutton',
                'comp_show_fmap_tbutton'))

        # Set the line edit cursor positions to zero
        self.set_zero_line_cursor((
            'plan_ledit', 'img_path_ledit', 'dose_path_ledit',
            'init_fluence_ledit', 'lower_var_ledit', 'upper_var_ledit',
            'ref_vol_ledit', 'ref_dose_ledit', 'display_segments_ledit'))

        # Set back to the arrow cursor
        QApplication.restoreOverrideCursor()

        return True

    def evaluate(self):
        """
        Evaluate the treatment plan.

        Returns
        -------
        bool
            Indicator for the success of the evaluation.
        """

        # Set the waiting cursor
        QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))

        try:

            # Get the treatment plan instance
            instance = self.plans[self.plan_ledit.text()]

            # Evaluate the treatment plan
            instance.evaluate()

            # Reset the DVH widget
            self.dvh_widget.reset_dvh()

            # Add the style and input data to the DVH widget
            self.dvh_widget.add_style_and_data(instance.datahub.dose_histogram)

            self.set_enabled(('visualize_pbutton',))

        except Exception as error:

            # Set back to the arrow cursor
            QApplication.restoreOverrideCursor()

            # Show a warning message box
            QMessageBox.warning(self, "pyanno4rt", str(error))

            return False

        # Update the plot of the DVH widget
        self.dvh_widget.update_dvh()

        # Update the output of the log window
        self.log_window.update_log_output()

        # Set the line edit cursor positions to zero
        self.set_zero_line_cursor((
            'plan_ledit', 'img_path_ledit', 'dose_path_ledit',
            'init_fluence_ledit', 'lower_var_ledit', 'upper_var_ledit',
            'ref_vol_ledit', 'ref_dose_ledit', 'display_segments_ledit'))

        # Set back to the arrow cursor
        QApplication.restoreOverrideCursor()

        return True

    def visualize(self):
        """
        Visualize the treatment plan.

        Returns
        -------
        bool
            Indicator for the success of the visualization.
        """

        try:

            # Visualize the treatment plan
            self.plans[self.plan_ledit.text()].visualize(parent=self)

        except Exception as error:

            # Show a warning message box
            QMessageBox.warning(self, "pyanno4rt", str(error))

            return False

        return True

    def update_configuration(self):
        """Update the configuration parameters."""

        try:

            # Overwrite the configuration dictionary of the current instance
            self.plans[self.plan_ledit.text()].update(
                self.transform_configuration_to_dict())

            # Disable specific fields
            self.set_disabled((
                'optimize_pbutton', 'evaluate_pbutton', 'visualize_pbutton'))

            # Set the line edit cursor positions to zero
            self.set_zero_line_cursor((
                'plan_ledit', 'img_path_ledit', 'dose_path_ledit'))

        except Exception as error:

            # Show a warning message box
            QMessageBox.warning(self, "pyanno4rt", str(error))

            return False

    def update_optimization(self):
        """Update the optimization parameters."""

        try:

            # Overwrite the optimization dictionary of the current instance
            self.plans[self.plan_ledit.text()].update(
                self.transform_optimization_to_dict())

            # Disable specific fields
            self.set_disabled(('evaluate_pbutton', 'visualize_pbutton'))

            # Set the line edit cursor positions to zero
            self.set_zero_line_cursor((
                'init_fluence_ledit', 'lower_var_ledit', 'upper_var_ledit'))

        except Exception as error:

            # Show a warning message box
            QMessageBox.warning(self, "pyanno4rt", str(error))

            return False

    def update_evaluation(self):
        """Update the evaluation parameters."""

        try:

            # Overwrite the evaluation dictionary of the current instance
            self.plans[self.plan_ledit.text()].update(
                self.transform_evaluation_to_dict())

            # Disable specific fields
            self.set_disabled(('visualize_pbutton',))

            # Set the line edit cursor positions to zero
            self.set_zero_line_cursor((
                'ref_vol_ledit', 'ref_dose_ledit', 'display_segments_ledit'))

        except Exception as error:

            # Show a warning message box
            QMessageBox.warning(self, "pyanno4rt", str(error))

            return False

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

        # Check if the target imaging resolution is specified
        if configuration['target_imaging_resolution']:

            # Set the target imaging resolution
            self.img_res_ledit.setText(
                str(configuration['target_imaging_resolution']))

        else:

            # Clear the target imaging resolution
            self.img_res_ledit.clear()

        # Set the dose matrix path
        self.dose_path_ledit.setText(abspath(
            configuration['dose_matrix_path']))

        # Set the dose resolution
        self.dose_res_ledit.setText(str(configuration['dose_resolution']))

        # Set the line edit cursor positions to zero
        self.set_zero_line_cursor((
            'plan_ledit', 'img_path_ledit', 'dose_path_ledit'))

    def set_optimization(self):
        """Set the optimization parameters."""

        def set_single_component(segment, component):
            """."""

            # Check if the component is an objective
            if component['type'] == 'objective':

                # Set the icon path on the red target
                icon_path = (":/special_icons/icons_special/"
                             "target-red-svgrepo-com.svg")

            else:

                # Set the icon path on the red frame
                icon_path = (":/special_icons/icons_special/"
                             "frame-red-svgrepo-com.svg")

            # Initialize the icon object
            icon = QIcon()

            # Add the pixmap to the icon
            icon.addPixmap(QPixmap(icon_path), QIcon.Normal, QIcon.Off)

            # Get the component parameters
            parameters = component['instance']['parameters']

            # Check if the identifier parameter is specified
            if 'identifier' in parameters:

                # Get the identifier string
                identifier = parameters['identifier']

            else:

                # Set the identifier to None
                identifier = None

            # Check if the embedding parameter is specified
            if 'embedding' in parameters:

                # Get the embedding string
                embedding = ''.join(('embedding: ',
                                     str(parameters['embedding'])))

            else:

                # Set the embedding string to the default
                embedding = 'embedding: active'

            # Check if the weight parameter is specified
            if 'weight' in parameters:

                # Get the weight string
                weight = ''.join(('weight: ',
                                  str(float(parameters['weight']))))

            else:

                # Set the weight string to the default
                weight = 'weight: 1'

            # Join the segment and class name
            component_string = ' - '.join((substring for substring in (
                segment, component['instance']['class'], identifier,
                embedding, weight) if substring))

            # Add the icon with the component string to the list
            self.components_lwidget.addItem(
                QListWidgetItem(icon, component_string))

            # Add the component to the components dictionary of the GUI
            self.plan_components[self.plan_ledit.text()][component_string] = (
                {segment: component})

        # Get the optimization dictionary
        optimization = self.plans[self.plan_ledit.text()].optimization

        # Clear the component list
        self.components_lwidget.clear()

        # Loop over the components
        for segment, component in optimization['components'].items():

            # Check if the component is a list
            if isinstance(component, list):

                # Loop over the component list
                for element in component:

                    # Set the component element
                    set_single_component(segment, element)

            else:

                # Set the component
                set_single_component(segment, component)

        # Set the optimization method
        self.method_cbox.setCurrentText(optimization['method'])

        # Set the solver package
        self.solver_cbox.setCurrentText(optimization['solver'])

        # Set the algorithm
        self.algorithm_cbox.setCurrentText(optimization['algorithm'])

        # Set the initialization strategy
        self.init_strat_cbox.setCurrentText(optimization['initial_strategy'])

        # Check if the initialization strategy is not 'warm-start'
        if self.init_strat_cbox.currentText() != 'warm-start':

            # Disable specific fields
            self.set_disabled(('init_fluence_ledit', 'init_fluence_tbutton',
                               'ref_plan_cbox'))

        else:

            # Enable specific fields
            self.set_enabled(('init_fluence_ledit', 'init_fluence_tbutton',
                              'ref_plan_cbox'))

        # Check if the initial fluence vector is specified
        if optimization['initial_fluence_vector']:

            # Set the initial fluence vector
            self.init_fluence_ledit.setText(
                str(optimization['initial_fluence_vector']))

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

        # Check if the upper variable bounds are specified
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

    def set_evaluation(self):
        """Set the evaluation parameters."""

        # Get the evaluation dictionary from the current plan
        evaluation = self.plans[self.plan_ledit.text()].evaluation

        # Set the DVH type
        self.dvh_type_cbox.setCurrentText(evaluation['dvh_type'])

        # Set the number of DVH points
        self.n_points_sbox.setValue(evaluation['number_of_points'])

        # Set the reference volume
        self.ref_vol_ledit.setText(
            '' if evaluation['reference_volume'] == [2, 5, 50, 95, 98]
            else str(evaluation['reference_volume']))

        # Set the reference dose values
        self.ref_dose_ledit.setText(
            '' if evaluation['reference_dose'] == []
            else str(evaluation['reference_dose']))

        # Set the display segments
        self.display_segments_ledit.setText(
            '' if evaluation['display_segments'] == []
            else str(evaluation['display_segments']).replace("\'", ''))

        # Loop over the display metrics
        for index in range(self.display_metrics_lwidget.count()):

            # Check if the display metric is included in the dictionary
            if (self.display_metrics_lwidget.item(index).text()
                    in evaluation['display_metrics']
                    or evaluation['display_metrics'] == []):

                # Set the display metric to checked
                self.display_metrics_lwidget.item(index).setCheckState(2)

            else:

                # Set the display metric to unchecked
                self.display_metrics_lwidget.item(index).setCheckState(0)

        # Set the line edit cursor positions to zero
        self.set_zero_line_cursor((
            'ref_vol_ledit', 'ref_dose_ledit', 'display_segments_ledit'))

    def clear_configuration(self):
        """Clear the configuration parameters."""

        # 
        if self.plan_ledit.text() not in (
                self.plan_select_cbox.itemText(i)
                for i in range(self.plan_select_cbox.count())):

            # Reset the treatment plan label
            self.plan_ledit.setText(self.base_configuration['label'])

            # 
            self.plan_ledit.setReadOnly(False)

        # Reset the minimum logging level
        self.log_level_cbox.setCurrentText(
            self.base_configuration['min_log_level'])

        # Reset the treatment modality
        self.modality_cbox.setCurrentText(self.base_configuration['modality'])

        # Reset the number of fractions
        self.nfx_sbox.setValue(self.base_configuration['number_of_fractions'])

        # Reset the imaging path
        self.img_path_ledit.setText(self.base_configuration['imaging_path'])

        # Clear the target imaging resolution
        self.img_res_ledit.clear()

        # Reset the dose matrix path
        self.dose_path_ledit.setText(
            self.base_configuration['dose_matrix_path'])

        # Reset the dose resolution
        self.dose_res_ledit.setText(self.base_configuration['dose_resolution'])

    def clear_optimization(self):
        """Clear the optimization parameters."""

        # Reset the components
        self.components_lwidget.clear()

        # Reset the optimization method
        self.method_cbox.setCurrentText(self.base_optimization['method'])

        # Reset the solver package
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
        self.tolerance_ledit.setText(str(self.base_optimization['tolerance']))

    def clear_evaluation(self):
        """Clear the evaluation parameters."""

        # Reset the DVH type
        self.dvh_type_cbox.setCurrentText(self.base_evaluation['dvh_type'])

        # Reset the number of DVH points
        self.n_points_sbox.setValue(self.base_evaluation['number_of_points'])

        # Reset the reference volume
        self.ref_vol_ledit.setText(
            str(self.base_evaluation['reference_volume']))

        # Reset the reference dose values
        self.ref_dose_ledit.setText(
            str(self.base_evaluation['reference_dose']))

        # Reset the display segments
        self.display_segments_ledit.setText(
            str(self.base_evaluation['display_segments']))

        # Loop over the display metrics
        for index in range(self.display_metrics_lwidget.count()):

            # Reset the display metric to checked
            self.display_metrics_lwidget.item(index).setCheckState(2)

    def transform_configuration_to_dict(self):
        """
        Transform the configuration fields into a dictionary.

        Returns
        -------
        dict
            Dictionary with the configuration parameters.
        """

        # Convert the target imaging resolution from the field
        target_imaging_resolution = add_square_brackets(
            self.img_res_ledit.text())

        # Convert the dose resolution from the field
        dose_resolution = add_square_brackets(self.dose_res_ledit.text())

        # Create the configuration dictionary
        configuration = {
            'label': None if not self.plan_ledit.text()
            else self.plan_ledit.text(),
            'min_log_level': self.log_level_cbox.currentText(),
            'modality': self.modality_cbox.currentText(),
            'number_of_fractions': self.nfx_sbox.value(),
            'imaging_path': None if not self.img_path_ledit.text()
            else abspath(self.img_path_ledit.text()),
            'target_imaging_resolution': None if not target_imaging_resolution
            else loads(target_imaging_resolution),
            'dose_matrix_path': None if not self.dose_path_ledit.text()
            else abspath(self.dose_path_ledit.text()),
            'dose_resolution': None if not dose_resolution
            else loads(dose_resolution)
            }

        return configuration

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

        # Loop over the components
        for component in self.plan_components.get(
                self.plan_ledit.text(), {}).values():

            # Get the key and value of the component
            (key, value), = component.items()

            # Check if the key is already included in the dictionary
            if key in components:

                # Create a list of values
                components[key] = [components[key], value]

            else:

                # Enter the value into the dictionary
                components[key] = value

        # Get the initial fluence sources
        sources = (self.init_fluence_ledit.text(),
                   self.ref_plan_cbox.currentText())

        # Convert the lower variable bounds from the field
        lower_variable_bounds = add_square_brackets(
            self.lower_var_ledit.text())

        # Convert the upper variable bounds from the field
        upper_variable_bounds = add_square_brackets(
            self.upper_var_ledit.text())

        # Create the optimization dictionary from the input fields
        optimization = {
            'components': components,
            'method': self.method_cbox.currentText(),
            'solver': self.solver_cbox.currentText(),
            'algorithm': self.algorithm_cbox.currentText(),
            'initial_strategy': self.init_strat_cbox.currentText(),
            'initial_fluence_vector': None if sources == ('', 'None')
            else loads(self.init_fluence_ledit.text()) if sources[0] != ''
            else self.plans[sources[1]].datahub.optimization[
                'optimized_fluence'].tolist(),
            'lower_variable_bounds': 0 if not lower_variable_bounds
            else loads(lower_variable_bounds),
            'upper_variable_bounds': None if not upper_variable_bounds
            else loads(upper_variable_bounds),
            'max_iter': self.max_iter_sbox.value(),
            'tolerance': 1e-3 if self.tolerance_ledit.text() == ''
            else loads(self.tolerance_ledit.text())
            }

        return optimization

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

        # Get the display segments text
        display_segments = self.display_segments_ledit.text()

        # Create the evaluation dictionary from the input fields
        evaluation = {
            'dvh_type': self.dvh_type_cbox.currentText(),
            'number_of_points': self.n_points_sbox.value(),
            'reference_volume': [2, 5, 50, 95, 98]
            if reference_volume in ('', '[]')
            else loads(reference_volume),
            'reference_dose': [] if reference_dose in ('', '[]')
            else loads(reference_dose),
            'display_segments': [] if display_segments in ('', '[]')
            else display_segments.strip('][').split(', '),
            'display_metrics': [
                self.display_metrics_lwidget.item(index).text()
                for index in range(self.display_metrics_lwidget.count())
                if self.display_metrics_lwidget.item(index).checkState()]
            }

        return evaluation

    def open_plan_creation_window(self):
        """Open the plan creation window."""

        # Set the position of the window
        self.plan_creation_window.position()

        # 
        if [self.plan_select_cbox.itemText(i) for i in range(
                self.plan_select_cbox.count())] != ['', 'Create new plan']:

            # 
            self.plan_creation_window.new_plan_ref_cbox.setEnabled(True)

        else:

            # 
            self.plan_creation_window.new_plan_ref_cbox.setEnabled(False)

        # 
        self.plan_creation_window.create_plan_pbutton.setEnabled(False)

        # 
        self.plan_creation_window.new_plan_ledit.clear()
        self.plan_creation_window.new_plan_ref_cbox.clear()
        self.plan_creation_window.new_img_path_ledit.clear()
        self.plan_creation_window.new_dose_path_ledit.clear()
        self.plan_creation_window.new_dose_res_ledit.clear()

        # 
        self.plan_creation_window.new_plan_ref_cbox.addItem('None')

        # 
        for plan in (self.plan_select_cbox.itemText(i)
                     for i in range(self.plan_select_cbox.count())):

            # 
            if plan not in ('', 'Create new plan'):

                # 
                self.plan_creation_window.new_plan_ref_cbox.addItem(plan)

        # Sort the items in the selector alphabetically
        self.plan_creation_window.new_plan_ref_cbox.model().sort(0)

        # Show the window
        self.plan_creation_window.show()

    def open_settings_window(self):
        """Open the settings window."""

        # Set the position of the window
        self.settings_window.position()

        # Set the zero item text to the current window size
        self.settings_window.resolution_cbox.setItemText(
            0, 'x'.join(map(str, (self.width(), self.height()))))

        # Set the current resolution combo box index to zero
        self.settings_window.resolution_cbox.setCurrentIndex(0)

        # Show the window
        self.settings_window.show()

    def open_info_window(self):
        """Open the information window."""

        # Set the position of the window
        self.info_window.position()

        # Show the window
        self.info_window.show()

    def open_compare_window(self):
        """Open the plan comparison window."""

        # 
        baseline = self.plans[self.baseline_cbox.currentText()]
        reference = self.plans[self.reference_cbox.currentText()]

        # 
        if (baseline.datahub.computed_tomography['cube'].shape ==
                reference.datahub.computed_tomography['cube'].shape):

            # 
            self.compare_window.baseline_label.setText(
                self.baseline_cbox.currentText())
            self.compare_window.reference_label.setText(
                self.reference_cbox.currentText())

            # 
            self.compare_window.baseline_slice_widget.reset_images()
            self.compare_window.reference_slice_widget.reset_images()

            # 
            self.compare_window.baseline_slice_widget.add_ct(
                baseline, baseline.datahub.computed_tomography['cube'])
            self.compare_window.reference_slice_widget.add_ct(
                reference, reference.datahub.computed_tomography['cube'])

            # Get the axial dimension of the CT cube
            axial_length = baseline.datahub.computed_tomography[
                'cube_dimensions'][2]

            # Add the segments to the slice widget
            self.compare_window.baseline_slice_widget.add_segments(
                baseline.datahub.computed_tomography,
                baseline.datahub.segmentation)
            self.compare_window.reference_slice_widget.add_segments(
                reference.datahub.computed_tomography,
                reference.datahub.segmentation)

            # Set the range of the slice selection scrollbar
            self.compare_window.slice_selection_sbar.setRange(
                0, axial_length-1)

            # Set the initial scrollbar value
            self.compare_window.slice_selection_sbar.setValue(
                int((axial_length-1)/2))

            # Set the initial position label
            self.compare_window.slice_selection_pos.setText(''.join((
                'z = ', str(self.compare_window.baseline_slice_widget.slice),
                ' mm')))

            # Get the joint minimum and maximum dose
            minima = [min(
                baseline.datahub.optimization[
                    'optimized_dose'][:, :, index].min(),
                reference.datahub.optimization[
                    'optimized_dose'][:, :, index].min())
                for index in range(axial_length)]
            maxima = [max(
                baseline.datahub.optimization[
                    'optimized_dose'][:, :, index].max(),
                reference.datahub.optimization[
                    'optimized_dose'][:, :, index].max())
                for index in range(axial_length)]

            # Add the dose image to the slice widget
            self.compare_window.baseline_slice_widget.add_dose(
                baseline.datahub.optimization['optimized_dose'],
                minima, maxima)
            self.compare_window.reference_slice_widget.add_dose(
                reference.datahub.optimization['optimized_dose'],
                minima, maxima)

            # 
            self.compare_window.baseline_slice_widget.update_images()
            self.compare_window.reference_slice_widget.update_images()

            # Set the position of the window
            self.compare_window.position()

            # Show the window
            self.compare_window.show()

        else:

            message = ("Baseline and reference treatment plan have different "
                       "CT cube dimensions. Please select treatment plans "
                       "with equal dimensions for comparison!")

            QMessageBox.information(self, 'pyanno4rt', message)

    def open_component_window(self, name):
        """Open a component window."""

        self.current_component_window = component_window_map[name](self)

        # Set the position of the window
        self.current_component_window.position()

        # Show the window
        self.current_component_window.show()

    def open_parameter_window(self):
        """Open the plan parameter window."""

        # Set the position of the window
        self.parameter_window.position()

        # Clear the parameter window
        self.parameter_window.tree_widget.clear()

        # Get the treatment plan instance
        instance = self.plans[self.plan_ledit.text()]

        # Loop over the configuration file paths
        for key in ('imaging_path', 'dose_matrix_path'):

            # Convert the input into absolute paths
            instance.configuration[key] = abspath(instance.configuration[key])

        # Create the parameter tree from the input dictionaries
        self.parameter_window.create_tree_from_dict(data={
            'configuration': instance.configuration,
            'optimization': instance.optimization,
            'evaluation': instance.evaluation},
            parent=self.parameter_window.tree_widget)

        # Set the resize mode for the first tree column
        self.parameter_window.tree_widget.header().setSectionResizeMode(
            0, QHeaderView.Stretch)

        # Show the window
        self.parameter_window.show()

    def open_plan_window(self):
        """Open the plan data window."""

        # Set the position of the window
        self.plan_window.position()

        # Clear the plan data window
        self.plan_window.tree_widget.clear()

        # Get the treatment plan instance
        instance = self.plans[self.plan_ledit.text()]

        # Create the plan data tree from the internal plan dictionaries
        self.plan_window.create_tree_from_dict(data={
            'computed_tomography': instance.datahub.computed_tomography,
            'segmentation': instance.datahub.segmentation,
            'plan_configuration': instance.datahub.plan_configuration,
            'dose_information': instance.datahub.dose_information,
            'optimization': instance.datahub.optimization,
            'dose_histogram': instance.datahub.dose_histogram,
            'dosimetrics': instance.datahub.dosimetrics},
            parent=self.plan_window.tree_widget)

        # Set the resize mode for the first tree column
        self.plan_window.tree_widget.header().setSectionResizeMode(
            0, QHeaderView.Stretch)

        # Show the window
        self.plan_window.show()

    def open_model_data_window(self):
        """Open the model data window."""

        # Set the position of the window
        self.model_data_window.position()

        # Clear the model data window
        self.model_data_window.tree_widget.clear()

        # Get the treatment plan instance
        instance = self.plans[self.plan_ledit.text()]

        # Create the model data tree from the internal model dictionaries
        self.model_data_window.create_tree_from_dict(data={
            'datasets': instance.datahub.datasets,
            'model_instances': instance.datahub.model_instances,
            'model_inspections': instance.datahub.model_inspections,
            'model_evaluations': instance.datahub.model_evaluations},
            parent=self.model_data_window.tree_widget)

        # Set the resize mode for the first tree column
        self.model_data_window.tree_widget.header().setSectionResizeMode(
            0, QHeaderView.Stretch)

        # Show the window
        self.model_data_window.show()

    def open_feature_map_window(self):
        """Open the feature map window."""

        # Set the position of the window
        self.feature_map_window.position()

        # Clear the feature map window
        self.feature_map_window.tree_widget.clear()

        # Create the feature map tree from the internal dictionary
        self.feature_map_window.create_tree_from_dict(
            data=self.plans[self.plan_ledit.text()].datahub.feature_maps,
            parent=self.feature_map_window.tree_widget)

        # Set the resize mode for the first tree column
        self.feature_map_window.tree_widget.header().setSectionResizeMode(
            0, QHeaderView.Stretch)

        # Show the window
        self.feature_map_window.show()

    def open_log_window(self):
        """Open the log window."""

        # Set the position of the window
        self.log_window.position()

        # Show the window
        self.log_window.show()

    def open_question_dialog(self):
        """Open a question dialog."""

        # Initialize the dictionary with the messages and function calls
        sources = {
            'drop_pbutton': (
                "Dropping the current treatment plan will irreversibly remove "
                "it from the GUI. Are you sure you want to proceed?",
                self.drop_tpi),
            'update_configuration_pbutton': (
                "Updating will change the configuration parameters of the "
                "current treatment plan. Are you sure you want to proceed?",
                self.update_configuration),
            'update_optimization_pbutton': (
                "Updating will change the optimization parameters of the "
                "current treatment plan. Are you sure you want to proceed?",
                self.update_optimization),
            'update_evaluation_pbutton': (
                "Updating will change the evaluation parameters of the "
                "current treatment plan. Are you sure you want to proceed?",
                self.update_evaluation),
            'reset_configuration_pbutton': (
                "Resetting will change all configuration tab fields to the "
                "last saved state. Are you sure you want to proceed?",
                self.set_configuration),
            'reset_optimization_pbutton': (
                "Resetting will change all optimization tab fields to the "
                "last saved state. Are you sure you want to proceed?",
                self.set_optimization),
            'reset_evaluation_pbutton': (
                "Resetting will change all evaluation tab fields to the last "
                "saved state. Are you sure you want to proceed?",
                self.set_evaluation),
            'clear_configuration_pbutton': (
                "Clearing will change all configuration tab fields to the "
                "default state. Are you sure you want to proceed?",
                self.clear_configuration),
            'clear_optimization_pbutton': (
                "Clearing will change all optimization tab fields to the "
                "default state. Are you sure you want to proceed?",
                self.clear_optimization),
            'clear_evaluation_pbutton': (
                "Clearing will change all evaluation tab fields to the "
                "default state. Are you sure you want to proceed?",
                self.clear_evaluation)
            }

        # Get the message and function call by the attribute argument
        message, call = sources[self.sender().objectName()]

        # Check if the question dialog is confirmed
        if (QMessageBox.question(self, 'pyanno4rt', message)
                == QMessageBox.Yes):

            # Call the function
            call()

    def add_imaging_path(self):
        """Add the CT and segmentation data from a folder."""

        # Get the file path
        path, _ = QFileDialog.getOpenFileName(
            self, 'Select a patient data file', QDir.rootPath(),
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
            self, 'Select a dose-influence matrix file', QDir.rootPath(),
            'Dose-influence matrix (*.mat *.npy)')

        # Check if the file path exists
        if path:

            # Set the dose matrix path field
            self.dose_path_ledit.setText(abspath(path))

            # Set the dose matrix path field cursor position to zero
            self.dose_path_ledit.setCursorPosition(0)

    def remove_component(self):
        """Remove the selected component from the instance."""

        # Remove the component from the GUI components dictionary
        del self.plan_components[self.plan_ledit.text()][
            self.components_lwidget.currentItem().text()]

        # Remove the item from the list widget
        self.components_lwidget.takeItem(self.components_lwidget.currentRow())

        # Clear the selection in the list widget
        self.components_lwidget.selectionModel().clear()

        # Disable specific fields
        self.set_disabled((
            'components_minus_tbutton', 'components_edit_tbutton'))

    def edit_component(self):
        """Open a component window."""

        # 
        component = self.plan_components[self.plan_ledit.text()][
            self.components_lwidget.currentItem().text()]

        # 
        for value in component.values():

            # 
            self.current_component_window = component_window_map[
                value['instance']['class']](self)

        # Set the position of the window
        self.current_component_window.position()

        # 
        self.current_component_window.load(component)

        # Show the window
        self.current_component_window.show()

    def add_initial_fluence_vector(self):
        """Add the initial fluence vector from a file."""

        # Get the file path
        path, _ = QFileDialog.getOpenFileName(
            self, 'Select an initial fluence vector file', QDir.rootPath(),
            'Fluence vector (*.json *.p *.txt)')

        # Check if the file path exists
        if path:

            # Get the list of values
            value_list = load_list_from_file(path)

            # Set the initial fluence vector field
            self.init_fluence_ledit.setText(str(value_list))

            # Set the initial fluence vector field cursor position to zero
            self.init_fluence_ledit.setCursorPosition(0)

    def add_lower_var_bounds(self):
        """Add the lower variable bounds from a file."""

        # Get the file path
        path, _ = QFileDialog.getOpenFileName(
            self, 'Select a lower variable bounds file', QDir.rootPath(),
            'Fluence vector (*.json *.p *.txt)')

        # Check if the file path exists
        if path:

            # Get the list of values
            value_list = load_list_from_file(path)

            # Set the lower variable bounds field
            self.lower_var_ledit.setText(str(value_list))

            # Set the lower variable bounds field cursor position to zero
            self.lower_var_ledit.setCursorPosition(0)

    def add_upper_var_bounds(self):
        """Add the upper variable bounds from a file."""

        # Get the file path
        path, _ = QFileDialog.getOpenFileName(
            self, 'Select an upper variable bounds file', QDir.rootPath(),
            'Fluence vector (*.json *.p *.txt)')

        # Check if the file path exists
        if path:

            # Get the list of values
            value_list = load_list_from_file(path)

            # Set the upper variable bounds field
            self.upper_var_ledit.setText(str(value_list))

            # Set the upper variable bounds field cursor position to zero
            self.upper_var_ledit.setCursorPosition(0)

    def update_init_button(self):
        """."""

        # 
        if self.plan_ledit.text() in self.plans:

            # 
            self.initialize_pbutton.setEnabled(False)

        else:

            if any(text == '' for text in (
                self.plan_ledit.text(), self.img_path_ledit.text(),
                    self.dose_path_ledit.text(), self.dose_res_ledit.text())):

                # 
                self.initialize_pbutton.setEnabled(False)

            else:

                # 
                self.initialize_pbutton.setEnabled(True)

    def update_by_initial_strategy(self):
        """Update the GUI by the initial strategy."""

        # Check if the initial strategy is not 'warm-start'
        if self.init_strat_cbox.currentText() != 'warm-start':

            # Disable specific fields
            self.set_disabled(('init_fluence_ledit', 'init_fluence_tbutton',
                               'ref_plan_cbox'))

        else:

            # Enable specific fields
            self.set_enabled(('init_fluence_ledit', 'init_fluence_tbutton',
                              'ref_plan_cbox'))

    def update_by_initial_fluence(self):
        """Update the GUI by the initial fluence vector."""

        # Check if an initial fluence vector is passed
        if self.init_fluence_ledit.text() != '':

            # Disable the reference plan combo box
            self.ref_plan_cbox.setEnabled(False)

        else:

            # Enable the reference plan combo box
            self.ref_plan_cbox.setEnabled(True)

    def update_by_reference(self):
        """Update the GUI by the reference plan."""

        # Check if a reference plan is passed
        if self.ref_plan_cbox.currentText() != 'None':

            # Disable specific fields
            self.set_disabled(('init_fluence_ledit', 'init_fluence_tbutton'))

        else:

            # Enable specific fields
            self.set_enabled(('init_fluence_ledit', 'init_fluence_tbutton'))

    def update_by_method(self):
        """Update the GUI by the optimization method."""

        # Clear the solver combo box
        self.solver_cbox.clear()

        # Check if the method is 'pareto'
        if self.method_cbox.currentText() == 'pareto':

            # Add the solver options to the combo box
            self.solver_cbox.addItems(['pymoo'])

            # Set the default solver option
            self.solver_cbox.setCurrentText('pymoo')

        else:

            # Add the solver options to the combo box
            self.solver_cbox.addItems(['proxmin', 'pypop7', 'scipy'])

            # Set the default solver option
            self.solver_cbox.setCurrentText('scipy')

    def update_by_solver(self):
        """Update the GUI by the solver."""

        # Clear the algorithm combo box
        self.algorithm_cbox.clear()

        # Check if the solver is 'proxmin'
        if self.solver_cbox.currentText() == 'proxmin':

            # Add the algorithm options to the combo box
            self.algorithm_cbox.addItems(['admm', 'pgm', 'sdmm'])

            # Set the default algorithm option
            self.algorithm_cbox.setCurrentText('pgm')

        # Else, check if the solver is 'pymoo'
        elif self.solver_cbox.currentText() == 'pymoo':

            # Add the algorithm options to the combo box
            self.algorithm_cbox.addItems(['NSGA3'])

            # Set the default algorithm option
            self.algorithm_cbox.setCurrentText('NSGA3')

        # Else, check if the solver is 'pypop7'
        elif self.solver_cbox.currentText() == 'pypop7':

            # Add the algorithm options to the combo box
            self.algorithm_cbox.addItems(['LMCMA', 'LMMAES'])

            # Set the default algorithm option
            self.algorithm_cbox.setCurrentText('LMMAES')

        # Else, check if the solver is 'scipy'
        elif self.solver_cbox.currentText() == 'scipy':

            # Add the algorithm options to the combo box
            self.algorithm_cbox.addItems(['L-BFGS-B', 'TNC', 'trust-constr'])

            # Set the default algorithm option
            self.algorithm_cbox.setCurrentText('L-BFGS-B')

    def update_reference_plans(self):
        """Update the available reference plans."""

        # Initialize the plans by the default 'None' plan
        plans = [str(None)]

        # Extend the plans by all non-selected, but optimized plans
        plans.extend([plan for plan in self.optimized_plans
                      if plan != self.plan_select_cbox.currentText()])

        # Clear the reference plan combo box
        self.ref_plan_cbox.clear()

        # Add the available reference plans to the combo box
        self.ref_plan_cbox.addItems(plans)

    def update_comparison_plans(self):
        """."""

        # Initialize the plans by the default empty label
        plans = ['']

        # Extend the plans by all optimized plans
        plans.extend(self.optimized_plans)

        # 
        self.baseline_cbox.clear()
        self.reference_cbox.clear()

        # 
        self.baseline_cbox.addItems(plans)
        self.reference_cbox.addItems(plans)

        # 
        if len(self.optimized_plans) > 1:

            # 
            self.baseline_cbox.setEnabled(True)
            self.reference_cbox.setEnabled(True)

        else:

            # 
            self.baseline_cbox.setEnabled(False)
            self.reference_cbox.setEnabled(False)

    def update_compare_button(self):
        """."""

        # 
        if (self.baseline_cbox.currentText()
                != self.reference_cbox.currentText()
                and all(text != '' for text in (
                    self.baseline_cbox.currentText(),
                    self.reference_cbox.currentText()))):

            # 
            self.compare_pbutton.setEnabled(True)

        else:

            # 
            self.compare_pbutton.setEnabled(False)

    def exit_window(self):
        """Exit the session and close the window."""

        # Close the window
        self.close()
