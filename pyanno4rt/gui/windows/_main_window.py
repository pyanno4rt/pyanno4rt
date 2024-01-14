"""Main window."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from os.path import dirname
from json import loads
from PyQt5.QtCore import QDir, QEvent, Qt
from PyQt5.QtGui import QCursor, QIcon, QPixmap
from PyQt5.QtWidgets import (QApplication, QComboBox, QFileDialog, QHeaderView,
                             QListWidgetItem, QMainWindow, QMessageBox,
                             QSpinBox)

# %% Internal package import

from pyanno4rt.base import TreatmentPlan
from pyanno4rt.gui.compilations.main_window import Ui_main_window
from pyanno4rt.gui.custom_widgets import DVHWidget, SliceWidget
from pyanno4rt.gui.windows import (InfoWindow, LogWindow, SettingsWindow,
                                   TreeWindow)
from pyanno4rt.tools import (copycat, load_list_from_file, make_list_string,
                             snapshot)

# %% Class definition


class MainWindow(QMainWindow, Ui_main_window):
    """
    Main window for the GUI.

    This class creates the main window for the graphical user interface, \
    including logo, labels, and input/control elements.

    Parameters
    ----------
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

        # Get the base dictionaries
        self.base_configuration = self.transform_configuration_to_dict()
        self.base_optimization = self.transform_optimization_to_dict()
        self.base_evaluation = self.transform_evaluation_to_dict()

        # Set the stylesheet for the input fields
        ledit_style = ('''
                       QLineEdit {
                           color: rgb(0, 0, 0);
                           background-color: rgb(238, 238, 236);
                           border: 1px solid;
                           border-color: rgb(186, 189, 182);
                           }
                       QLineEdit:disabled {
                           color: rgb(153, 153, 153);
                           }
                       ''')

        # Set the stylesheet for the input fields
        cbox_style = ('''
                      QComboBox {
                          color: rgb(0, 0, 0);
                          background-color: rgb(238, 238, 236);
                          border: 1px solid;
                          border-color: rgb(186, 189, 182);
                          }
                      QComboBox QAbstractItemView {
                          color: rgb(0, 0, 0);
                          background-color: rgb(238, 238, 236);
                          }
                      QComboBox:disabled {
                          color: rgb(153, 153, 153);
                          }
                      ''')

        # 
        self.init_fluence_ledit.setStyleSheet(ledit_style)
        self.ref_plan_cbox.setStyleSheet(cbox_style)

        # Set the stylesheet for the menu buttons
        qbutton_style_menu = ('''
                              QPushButton {
                                  background-color: rgb(211, 215, 207);
                                  color: rgb(0, 0, 0);
                                  border: 1px solid;
                                  border-color: rgb(186, 189, 182);
                                  }
                              QPushButton:disabled {
                                  color: rgb(153, 153, 153);
                                  }
                              QPushButton:hover {
                                  background-color: rgb(238, 238, 236);
                                  }
                              ''')

        # Set the stylesheet for the composer tool buttons
        tbutton_style_composer = ('''
                                  QToolButton {
                                      color: rgb(0, 0, 0);
                                      background-color: rgb(238, 238, 236);
                                      border: 1px solid;
                                      border-color: rgb(186, 189, 182);
                                      }
                                  QToolButton:disabled {
                                      color: rgb(153, 153, 153);
                                      }
                                  QToolButton:hover {
                                      background-color: rgb(246, 246, 245);
                                      }
                                  ''')

        # Set the stylesheet for the composer update/reset buttons
        qbutton_style_composer = ('''
                                  QPushButton {
                                      color: rgb(0, 0, 0);
                                      background-color: rgb(238, 238, 236);
                                      border: 1px solid;
                                      border-color: rgb(186, 189, 182);
                                      }
                                  QPushButton:disabled {
                                      color: rgb(153, 153, 153);
                                      }
                                  QPushButton:hover {
                                      background-color: rgb(246, 246, 245);
                                      }
                                  ''')

        # Set the stylesheet for the workflow push buttons
        qbutton_style_workflow = ('''
                                  QPushButton {
                                      color: rgb(0, 0, 0);
                                      background-color: rgb(211, 215, 207);
                                      border: 1px solid;
                                      border-color: rgb(186, 189, 182);
                                      }
                                  QPushButton:disabled {
                                      color: rgb(153, 153, 153);
                                      }
                                  QPushButton:hover {
                                      background-color: rgb(238, 238, 236);
                                      }
                                  ''')

        # Set the stylesheet for the workflow tool buttons
        tbutton_style_workflow = ('''
                                  QToolButton {
                                      color: rgb(0, 0, 0);
                                      background-color: rgb(211, 215, 207);
                                      border: 1px solid;
                                      border-color: rgb(186, 189, 182);
                                      }
                                  QToolButton:disabled {
                                      color: rgb(153, 153, 153);
                                      }
                                  QToolButton:hover {
                                      background-color: rgb(238, 238, 236);
                                      }
                                  ''')

        # 
        self.load_pbutton.setStyleSheet(qbutton_style_menu)
        self.save_pbutton.setStyleSheet(qbutton_style_menu)
        self.drop_pbutton.setStyleSheet(qbutton_style_menu)
        self.settings_pbutton.setStyleSheet(qbutton_style_menu)
        self.info_pbutton.setStyleSheet(qbutton_style_menu)
        self.exit_pbutton.setStyleSheet(qbutton_style_menu)

        # 
        self.img_path_tbutton.setStyleSheet(tbutton_style_composer)
        self.dose_path_tbutton.setStyleSheet(tbutton_style_composer)
        self.components_plus_tbutton.setStyleSheet(tbutton_style_composer)
        self.components_minus_tbutton.setStyleSheet(tbutton_style_composer)
        self.components_edit_tbutton.setStyleSheet(tbutton_style_composer)
        self.init_fluence_tbutton.setStyleSheet(tbutton_style_composer)
        self.lower_var_tbutton.setStyleSheet(tbutton_style_composer)
        self.upper_var_tbutton.setStyleSheet(tbutton_style_composer)

        # 
        self.update_configuration_pbutton.setStyleSheet(qbutton_style_composer)
        self.update_optimization_pbutton.setStyleSheet(qbutton_style_composer)
        self.update_evaluation_pbutton.setStyleSheet(qbutton_style_composer)

        # 
        self.reset_configuration_pbutton.setStyleSheet(qbutton_style_composer)
        self.reset_optimization_pbutton.setStyleSheet(qbutton_style_composer)
        self.reset_evaluation_pbutton.setStyleSheet(qbutton_style_composer)

        # 
        self.clear_configuration_pbutton.setStyleSheet(qbutton_style_composer)
        self.clear_optimization_pbutton.setStyleSheet(qbutton_style_composer)
        self.clear_evaluation_pbutton.setStyleSheet(qbutton_style_composer)

        # 
        self.initialize_pbutton.setStyleSheet(qbutton_style_workflow)
        self.configure_pbutton.setStyleSheet(qbutton_style_workflow)
        self.optimize_pbutton.setStyleSheet(qbutton_style_workflow)
        self.evaluate_pbutton.setStyleSheet(qbutton_style_workflow)
        self.visualize_pbutton.setStyleSheet(qbutton_style_workflow)

        # 
        self.show_parameter_tbutton.setStyleSheet(tbutton_style_workflow)
        self.show_plan_tbutton.setStyleSheet(tbutton_style_workflow)
        self.show_model_data_tbutton.setStyleSheet(tbutton_style_workflow)
        self.show_fmap_tbutton.setStyleSheet(tbutton_style_workflow)
        self.show_log_tbutton.setStyleSheet(tbutton_style_workflow)

        # Disable the menu buttons
        self.save_pbutton.setEnabled(False)
        self.drop_pbutton.setEnabled(False)

        # Disable the update buttons initially
        self.update_configuration_pbutton.setEnabled(False)
        self.update_optimization_pbutton.setEnabled(False)
        self.update_evaluation_pbutton.setEnabled(False)

        # Disable the reset buttons initially
        self.reset_configuration_pbutton.setEnabled(False)
        self.reset_optimization_pbutton.setEnabled(False)
        self.reset_evaluation_pbutton.setEnabled(False)

        # Disable the workflow buttons
        self.initialize_pbutton.setEnabled(False)
        self.configure_pbutton.setEnabled(False)
        self.optimize_pbutton.setEnabled(False)
        self.evaluate_pbutton.setEnabled(False)
        self.visualize_pbutton.setEnabled(False)

        # Disable the toolbox buttons
        self.show_parameter_tbutton.setEnabled(False)
        self.show_plan_tbutton.setEnabled(False)
        self.show_fmap_tbutton.setEnabled(False)
        self.show_model_data_tbutton.setEnabled(False)
        self.show_log_tbutton.setEnabled(False)

        # 
        self.init_fluence_ledit.setEnabled(False)
        self.init_fluence_tbutton.setEnabled(False)
        self.ref_plan_cbox.setEnabled(False)

        # Initialize the slice widget
        self.slice_widget = SliceWidget(self)

        # Insert the slice widget into the viewer
        self.tab_slices_layout.insertWidget(0, self.slice_widget)

        # Initialize the DVH widget
        self.dvh_widget = DVHWidget(self)

        # Insert the slice widget into the viewer
        self.tab_dvh_layout.insertWidget(0, self.dvh_widget)

        # Initialize the settings window
        self.settings_window = SettingsWindow(self)

        # Initialize the information window
        self.info_window = InfoWindow(self)

        # 
        self.parameter_window = TreeWindow('Plan parameters', self)

        # 
        self.plan_window = TreeWindow('Plan data', self)

        # 
        self.model_data_window = TreeWindow('Model data', self)

        # 
        self.feature_map_window = TreeWindow('Feature maps', self)

        # Initialize the logging window
        self.log_window = LogWindow(self)

        # Set the tab indices to ensure consistency of the starting window
        self.composer_widget.setCurrentIndex(0)
        self.tab_workflow.setCurrentIndex(0)
        self.viewer_widget.setCurrentIndex(0)

        # 
        tab_style = (
            """
            QTabBar::tab:selected
            {
                background-color: rgb(25, 25, 25);
            }
            """)

        # 
        self.composer_widget.tabBar().setStyleSheet(tab_style)
        self.tab_workflow.tabBar().setStyleSheet(tab_style)
        self.viewer_widget.tabBar().setStyleSheet(tab_style)

        # Connect the event signals
        self.connect_signals()

        # Disable the wheel events on QComboBox and QSpinBox
        self.log_level_cbox.installEventFilter(self)
        self.modality_cbox.installEventFilter(self)
        self.nfx_sbox.installEventFilter(self)
        self.method_cbox.installEventFilter(self)
        self.solver_cbox.installEventFilter(self)
        self.algorithm_cbox.installEventFilter(self)
        self.init_strat_cbox.installEventFilter(self)
        self.ref_plan_cbox.installEventFilter(self)
        self.max_iter_sbox.installEventFilter(self)
        self.dvh_type_cbox.installEventFilter(self)
        self.n_points_sbox.installEventFilter(self)
        self.opacity_sbox.installEventFilter(self)

        # Check if an initial treatment plan has been specified
        if treatment_plan:

            # 
            if isinstance(treatment_plan, TreatmentPlan):

                # Activate the treatment plan in the GUI
                self.activate(treatment_plan)

            # 
            elif isinstance(treatment_plan, list):

                # 
                for plan in treatment_plan:

                    # Activate the treatment plan in the GUI
                    self.activate(plan)

        # Set the initial window size
        self.resize(1920, 1080)

        # Show the GUI
        self.show()

    def eventFilter(self, source, event):
        """."""

        # 
        if (event.type() == QEvent.Wheel and
                isinstance(source, (QComboBox, QSpinBox))):

            # 
            return True

        # 
        return super().eventFilter(source, event)

    def connect_signals(self):
        """Connect the event signals to the GUI elements."""

        # Connect the plan selector
        self.plan_select_cbox.currentIndexChanged.connect(
            self.update_reference_plans)

        # Connect the menu buttons
        self.load_pbutton.clicked.connect(self.load_tpi)
        self.save_pbutton.clicked.connect(self.save_tpi)
        self.drop_pbutton.clicked.connect(self.open_drop_tpi_dialog)
        self.plan_select_cbox.currentTextChanged.connect(self.select_plan)
        self.settings_pbutton.clicked.connect(self.open_settings_window)
        self.info_pbutton.clicked.connect(self.open_info_window)
        self.exit_pbutton.clicked.connect(self.exit_window)

        # Connect the workflow buttons
        self.initialize_pbutton.clicked.connect(self.initialize)
        self.configure_pbutton.clicked.connect(self.configure)
        self.optimize_pbutton.clicked.connect(self.optimize)
        self.evaluate_pbutton.clicked.connect(self.evaluate)
        self.visualize_pbutton.clicked.connect(self.visualize)

        # Connect the toolbox buttons
        self.show_parameter_tbutton.clicked.connect(self.open_parameter_window)
        self.show_plan_tbutton.clicked.connect(self.open_plan_window)
        self.show_model_data_tbutton.clicked.connect(
            self.open_model_data_window)
        self.show_fmap_tbutton.clicked.connect(self.open_feature_map_window)
        self.show_log_tbutton.clicked.connect(self.open_log_window)

        # Connect the configuration buttons
        self.img_path_tbutton.clicked.connect(self.add_imaging_path)
        self.dose_path_tbutton.clicked.connect(self.add_dose_matrix_path)

        # Connect the optimization buttons
        self.init_fluence_tbutton.clicked.connect(
            self.add_initial_fluence_vector)
        self.lower_var_tbutton.clicked.connect(
            self.add_lower_var_bounds)
        self.upper_var_tbutton.clicked.connect(
            self.add_upper_var_bounds)

        # Connect the configuration field dependencies
        self.plan_ledit.textChanged.connect(self.update_by_plan_label)

        # Connect the optimization field dependencies
        self.method_cbox.currentIndexChanged.connect(self.update_by_method)
        self.solver_cbox.currentIndexChanged.connect(self.update_by_solver)
        self.init_strat_cbox.currentTextChanged.connect(
            self.update_by_initial_strategy)
        self.init_fluence_ledit.textChanged.connect(
            self.update_by_initial_fluence)
        self.ref_plan_cbox.currentIndexChanged.connect(
            self.update_by_reference)

        # Connect the update buttons
        self.update_configuration_pbutton.clicked.connect(
            self.open_update_configuration_dialog)
        self.update_optimization_pbutton.clicked.connect(
            self.open_update_optimization_dialog)
        self.update_evaluation_pbutton.clicked.connect(
            self.open_update_evaluation_dialog)

        # Connect the reset buttons
        self.reset_configuration_pbutton.clicked.connect(
            self.open_reset_configuration_dialog)
        self.reset_optimization_pbutton.clicked.connect(
            self.open_reset_optimization_dialog)
        self.reset_evaluation_pbutton.clicked.connect(
            self.open_reset_evaluation_dialog)

        # Connect the clear buttons
        self.clear_configuration_pbutton.clicked.connect(
            self.open_clear_configuration_dialog)
        self.clear_optimization_pbutton.clicked.connect(
            self.open_clear_optimization_dialog)
        self.clear_evaluation_pbutton.clicked.connect(
            self.open_clear_evaluation_dialog)

        # Connect the viewer elements
        self.opacity_sbox.valueChanged.connect(
            self.slice_widget.change_dose_opacity)
        self.slice_selection_sbar.valueChanged.connect(
            self.slice_widget.change_image_slice)

    def activate(self, plan):
        """."""

        # Add the instance to the gui plans dictionary
        self.plans[plan.configuration['label']] = plan

        # Check if the loaded instance does not yet appear in the selector
        if plan.configuration['label'] not in set(
                [self.plan_select_cbox.itemText(i)
                 for i in range(self.plan_select_cbox.count())]):

            # Add the loaded instance to the selector
            self.plan_select_cbox.addItem(plan.configuration['label'])

            # Select the loaded instance
            self.plan_select_cbox.setCurrentText(plan.configuration['label'])

            # Sort the items in the selector alphabetically
            self.plan_select_cbox.model().sort(0)

    def load_tpi(self):
        """Load the treatment plan from a snapshot folder."""

        # Get the loading folder path
        path = QFileDialog.getExistingDirectory(
            self, 'Select a directory for loading')

        # Check if the folder name exists (user has not stopped the loading)
        if path:

            # Apply the copycat function to build a treatment plan instance
            tp_copy = copycat(TreatmentPlan, path)

            # 
            self.activate(tp_copy)

    def save_tpi(self):
        """Save the treatment plan to a snapshot folder."""

        # Get the saving folder path
        path = QFileDialog.getExistingDirectory(
            self, 'Select a directory for saving')

        # Check if the folder name exists (user has not stopped the saving)
        if path:

            includes = self.settings_window.current[3]

            # Apply the snapshot function to save the treatment plan instance
            snapshot(self.plans[self.plan_ledit.text()],
                     ''.join((path, '/')), *includes)

    def drop_tpi(self):
        """Remove the current treatment plan."""

        # Check if the treatment plan is a validated instance
        if self.plan_ledit.text() in (*self.plans,):

            # Delete the instance from the gui plan dictionary
            del self.plans[self.plan_ledit.text()]

            # Check if the treatment plan has been optimized
            if self.plan_ledit.text() in self.optimized_plans:

                # Remove the label from the optimized plans list
                self.optimized_plans.remove(self.plan_ledit.text())

            # Remove the instance item from the selector
            self.plan_select_cbox.removeItem(
                self.plan_select_cbox.currentIndex())

            # Reset the selector index to the default (empty)
            # Note: this will automatically reset all input tab fields
            self.plan_select_cbox.setCurrentIndex(-1)

    def select_plan(self):
        """Select a treatment plan."""

        # 
        selection = self.plan_select_cbox.currentText()

        # Check if the selector text is not set to default
        if selection != '':

            # Change the treatment plan label to the instance label
            self.plan_ledit.setText(selection)

            # 
            label = self.plan_ledit.text()

            # Set the configuration, optimization and evaluation tab fields
            self.set_configuration()
            self.set_optimization()
            self.set_evaluation()

            # Enable the menu buttons
            self.save_pbutton.setEnabled(True)
            self.drop_pbutton.setEnabled(True)

            # Enable the update buttons
            self.update_configuration_pbutton.setEnabled(True)
            self.update_optimization_pbutton.setEnabled(True)
            self.update_evaluation_pbutton.setEnabled(True)

            # Enable the update buttons
            self.reset_configuration_pbutton.setEnabled(True)
            self.reset_optimization_pbutton.setEnabled(True)
            self.reset_evaluation_pbutton.setEnabled(True)

            # Disable the workflow buttons
            self.optimize_pbutton.setEnabled(False)
            self.evaluate_pbutton.setEnabled(False)
            self.visualize_pbutton.setEnabled(False)

            # 
            self.initialize_pbutton.setEnabled(False)

            # 
            self.show_parameter_tbutton.setEnabled(True)

            # 
            self.slice_widget.reset_images()

            # 
            self.dvh_widget.reset_dvh()

            # 
            if self.plans[label].input_checker:

                # 
                self.configure_pbutton.setEnabled(True)

            # 
            if (self.plans[label].datahub and all(isinstance(getattr(
                    self.plans[label].datahub, unit), dict)
                    for unit in ('computed_tomography', 'segmentation'))):

                # Get the CT dictionary
                computed_tomography = (
                    self.plans[label].datahub.computed_tomography)

                # Get the segmentation dictionary
                segmentation = self.plans[label].datahub.segmentation

                # Set the scrollbar range
                self.slice_selection_sbar.setRange(
                    0, computed_tomography['cube_dimensions'][2]-1)

                # Set the scrollbar value
                self.slice_selection_sbar.setValue(
                    int((computed_tomography['cube_dimensions'][2]-1)/2))

                # Add the CT image to the widget
                self.slice_widget.add_ct(computed_tomography['cube'])

                # Add the segments to the widget
                self.slice_widget.add_segments(computed_tomography,
                                               segmentation)

                # 
                self.optimize_pbutton.setEnabled(True)

                # 
                if isinstance(self.plans[label].datahub.optimization, dict):

                    # Get the optimized dose array
                    optimized_dose = self.plans[label].datahub.optimization[
                        'optimized_dose']

                    # Add the dose image to the widget
                    self.slice_widget.add_dose(optimized_dose)

                    # 
                    self.evaluate_pbutton.setEnabled(True)

                    # 
                    self.visualize_pbutton.setEnabled(True)

                # Update the slice image
                self.slice_widget.update_images()

            # 
            if (self.plans[label].datahub and isinstance(
                    self.plans[label].datahub.histogram,
                    dict)):

                # 
                self.dvh_widget.add_style_and_data(
                    self.plans[label].datahub.histogram)

                # 
                self.dvh_widget.update_dvh()

            # 
            if self.plans[label].datahub and self.plans[label].datahub.logger:

                # 
                self.show_plan_tbutton.setEnabled(True)

                # Update the log output
                self.log_window.update_log_output()

                # Enable the log window button
                self.show_log_tbutton.setEnabled(True)

            else:

                # 
                self.show_plan_tbutton.setEnabled(False)

                # 
                self.show_log_tbutton.setEnabled(False)

            # 
            if self.plans[label].datahub and all((
                    self.plans[label].datahub.datasets,
                    self.plans[label].datahub.model_instances,
                    self.plans[label].datahub.model_inspections,
                    self.plans[label].datahub.model_evaluations)):

                # 
                self.show_model_data_tbutton.setEnabled(True)

            else:

                # 
                self.show_model_data_tbutton.setEnabled(False)

            # 
            if (self.plans[label].datahub and
                    self.plans[label].datahub.feature_maps):

                # 
                self.show_fmap_tbutton.setEnabled(True)

            else:

                # 
                self.show_fmap_tbutton.setEnabled(False)

        else:

            # Change the treatment plan label to the default (empty)
            self.plan_ledit.setText('')

            # Clear the configuration, optimization and evaluation tab fields
            self.clear_configuration()
            self.clear_optimization()
            self.clear_evaluation()

            # Disable the menu buttons
            self.save_pbutton.setEnabled(False)
            self.drop_pbutton.setEnabled(False)

            # Disable the update buttons
            self.update_configuration_pbutton.setEnabled(False)
            self.update_optimization_pbutton.setEnabled(False)
            self.update_evaluation_pbutton.setEnabled(False)

            # Disable the reset buttons
            self.reset_configuration_pbutton.setEnabled(False)
            self.reset_optimization_pbutton.setEnabled(False)
            self.reset_evaluation_pbutton.setEnabled(False)

            # Disable the workflow buttons
            self.configure_pbutton.setEnabled(False)
            self.optimize_pbutton.setEnabled(False)
            self.evaluate_pbutton.setEnabled(False)
            self.visualize_pbutton.setEnabled(False)

            # Disable the toolbox buttons
            self.show_parameter_tbutton.setEnabled(False)
            self.show_plan_tbutton.setEnabled(False)
            self.show_model_data_tbutton.setEnabled(False)
            self.show_fmap_tbutton.setEnabled(False)
            self.show_log_tbutton.setEnabled(False)

            # 
            self.show_parameter_tbutton.setEnabled(False)

            # Reset the slice image
            self.slice_widget.reset_images()

            # Reset the DVH plot
            self.dvh_widget.reset_dvh()

    def open_settings_window(self):
        """Open the settings window."""

        # 
        self.settings_window.position()

        # 
        self.settings_window.resolution_cbox.setItemText(
            0, 'x'.join(map(str, (self.width(), self.height()))))

        # 
        self.settings_window.resolution_cbox.setCurrentIndex(0)

        # 
        self.settings_window.show()

    def open_info_window(self):
        """Open the information window."""

        # 
        self.info_window.position()

        # 
        self.info_window.show()

    def exit_window(self):
        """Exit the session and close the window."""

        # 
        self.settings_window.close()

        # 
        self.info_window.close()

        # Close the window
        self.close()

    def initialize(self):
        """Initialize the treatment plan."""

        # Set the waiting cursor
        QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))

        # Try to initialize the plan
        try:

            # Initialize the base class and add the instance to the dictionary
            self.plans[self.plan_ledit.text()] = TreatmentPlan(
                self.transform_configuration_to_dict(),
                self.transform_optimization_to_dict(),
                self.transform_evaluation_to_dict())

            # Add the instance to the selector
            self.plan_select_cbox.addItem(self.plan_ledit.text())

            # Change the selector text to the configured instance
            self.plan_select_cbox.setCurrentText(self.plan_ledit.text())

            # Sort the items in the selector alphabetically
            self.plan_select_cbox.model().sort(0)

            # 
            self.show_parameter_tbutton.setEnabled(True)

            # Reset back to the arrow cursor
            QApplication.restoreOverrideCursor()

            return True

        except Exception as error:

            # Reset back to the arrow cursor
            QApplication.restoreOverrideCursor()

            # 
            QMessageBox.warning(self, "pyanno4rt", str(error))

            return False

    def configure(self):
        """Configure the treatment plan."""

        # Set the waiting cursor
        QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))

        # Try to configure the plan
        try:

            # 
            instance = self.plans[self.plan_ledit.text()]

            # Run the configure method of the instance
            instance.configure()

            # Get the CT dictionary
            computed_tomography = instance.datahub.computed_tomography

            # Get the segmentation dictionary
            segmentation = instance.datahub.segmentation

            # Reset the slice image
            self.slice_widget.reset_images()

            # Set the scrollbar range
            self.slice_selection_sbar.setRange(
                0, computed_tomography['cube_dimensions'][2]-1)

            # Set the scrollbar value
            self.slice_selection_sbar.setValue(
                int((computed_tomography['cube_dimensions'][2]-1)/2))

            # Add the CT image to the widget
            self.slice_widget.add_ct(computed_tomography['cube'])

            # Add the segments to the widget
            self.slice_widget.add_segments(computed_tomography, segmentation)

            # Update the slice image
            self.slice_widget.update_images()

            # Enable the optimization button
            self.optimize_pbutton.setEnabled(True)

            # Enable the parameter window button
            self.show_parameter_tbutton.setEnabled(True)

            # Enable the plan window button
            self.show_plan_tbutton.setEnabled(True)

            # Update the log output
            self.log_window.update_log_output()

            # Enable the log window button
            self.show_log_tbutton.setEnabled(True)

            # Reset back to the arrow cursor
            QApplication.restoreOverrideCursor()

            return True

        except Exception as error:

            # Reset back to the arrow cursor
            QApplication.restoreOverrideCursor()

            # 
            QMessageBox.warning(self, "pyanno4rt", str(error))

            return False

    def optimize(self):
        """Optimize the treatment plan."""

        # Set the waiting cursor
        QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))

        try:

            # 
            instance = self.plans[self.plan_ledit.text()]

            # Run the optimize method of the instance
            instance.optimize()

            # Append the current treatment plan to the optimized plans list
            self.optimized_plans.append(self.plan_ledit.text())

            # Get the optimized dose array
            optimized_dose = instance.datahub.optimization['optimized_dose']

            # Add the dose image to the widget
            self.slice_widget.add_dose(optimized_dose)

            # Update the slice image
            self.slice_widget.update_images()

            # 
            if all((instance.datahub,
                    instance.datahub.datasets,
                    instance.datahub.model_instances,
                    instance.datahub.model_inspections,
                    instance.datahub.model_evaluations)):

                # 
                self.show_model_data_tbutton.setEnabled(True)

            # 
            if all((instance.datahub,
                    instance.datahub.feature_maps)):

                # 
                self.show_fmap_tbutton.setEnabled(True)

            # Update the log output
            self.log_window.update_log_output()

            # Enable the evaluation button
            self.evaluate_pbutton.setEnabled(True)

            # Enable the visualization button
            self.visualize_pbutton.setEnabled(True)

            # Reset back to the arrow cursor
            QApplication.restoreOverrideCursor()

        except Exception as error:

            # Reset back to the arrow cursor
            QApplication.restoreOverrideCursor()

            # 
            QMessageBox.warning(self, "pyanno4rt", str(error))

    def evaluate(self):
        """Evaluate the treatment plan."""

        # Set the waiting cursor
        QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))

        try:

            # 
            instance = self.plans[self.plan_ledit.text()]

            # Run the evaluate method of the instance
            instance.evaluate()

            # 
            self.dvh_widget.reset_dvh()

            # 
            histogram = instance.datahub.histogram

            # 
            self.dvh_widget.add_style_and_data(histogram)

            # 
            self.dvh_widget.update_dvh()

            # Update the log output
            self.log_window.update_log_output()

            # Reset back to the arrow cursor
            QApplication.restoreOverrideCursor()

        except Exception as error:

            # Reset back to the arrow cursor
            QApplication.restoreOverrideCursor()

            # 
            QMessageBox.warning(self, "pyanno4rt", str(error))

    def visualize(self):
        """Visualize the treatment plan."""

        try:

            # Run the visualize method of the instance
            self.plans[self.plan_ledit.text()].visualize(parent=self)

        except Exception as error:

            # 
            QMessageBox.warning(self, "pyanno4rt", str(error))

    def open_parameter_window(self):
        """Open the plan parameters window."""

        # 
        instance = self.plans[self.plan_ledit.text()]

        # 
        self.parameter_window.tree_widget.clear()

        # 
        self.parameter_window.position()

        # 
        self.parameter_window.create_tree_from_dict(
            data={
                'configuration': instance.configuration,
                'optimization': instance.optimization,
                'evaluation': instance.evaluation
                },
            parent=self.parameter_window.tree_widget)

        # 
        self.parameter_window.tree_widget.header().setSectionResizeMode(
            0, QHeaderView.Stretch)

        # 
        self.parameter_window.show()

    def open_plan_window(self):
        """Open the plan data window."""

        # 
        instance = self.plans[self.plan_ledit.text()]

        # 
        self.plan_window.tree_widget.clear()

        # 
        self.plan_window.position()

        # 
        self.plan_window.create_tree_from_dict(
            data={
                'computed_tomography': instance.datahub.computed_tomography,
                'segmentation': instance.datahub.segmentation,
                'plan_configuration': instance.datahub.plan_configuration,
                'dose_information': instance.datahub.dose_information,
                'optimization': instance.datahub.optimization,
                'histogram': instance.datahub.histogram,
                'dosimetrics': instance.datahub.dosimetrics
                },
            parent=self.plan_window.tree_widget)

        # 
        self.plan_window.tree_widget.header().setSectionResizeMode(
            0, QHeaderView.Stretch)
        self.plan_window.tree_widget.header().setSectionResizeMode(
            1, QHeaderView.Stretch)

        # 
        self.plan_window.show()

    def open_model_data_window(self):
        """Open the model data window."""

        # 
        instance = self.plans[self.plan_ledit.text()]

        # 
        self.model_data_window.tree_widget.clear()

        # 
        self.model_data_window.position()

        # 
        self.model_data_window.create_tree_from_dict(
            data={
                'datasets': instance.datahub.datasets,
                'model_instances': instance.datahub.model_instances,
                'model_inspections': instance.datahub.model_inspections,
                'model_evaluations': instance.datahub.model_evaluations
                },
            parent=self.model_data_window.tree_widget)

        # 
        self.model_data_window.tree_widget.header().setSectionResizeMode(
            0, QHeaderView.Stretch)

        # 
        self.model_data_window.show()

    def open_feature_map_window(self):
        """Open the feature map window."""

        # 
        instance = self.plans[self.plan_ledit.text()]

        # 
        self.feature_map_window.tree_widget.clear()

        # 
        self.feature_map_window.position()

        # 
        self.feature_map_window.create_tree_from_dict(
            data=instance.datahub.feature_maps,
            parent=self.feature_map_window.tree_widget)

        # 
        self.feature_map_window.tree_widget.header().setSectionResizeMode(
            0, QHeaderView.Stretch)

        # 
        self.feature_map_window.show()

    def open_log_window(self):
        """Open the log window."""

        # 
        self.log_window.position()

        # 
        self.log_window.show()

    def add_imaging_path(self):
        """Add the CT and segmentation data from a folder."""

        # Get the file path
        path, _ = QFileDialog.getOpenFileName(
            self, 'Select a patient data file', QDir.rootPath(),
            'CT/Segmentation data (*.dcm *.mat *.p)')

        # Check if the file path exists (user has not stopped the adding)
        if path:

            # Check if a DICOM file is passed
            if path.endswith('.dcm'):

                # Get the directory path of the file
                path = dirname(path)

            # Set the imaging path field to the path
            self.img_path_ledit.setText(path)

    def add_dose_matrix_path(self):
        """Add the dose-influence matrix from a folder."""

        # Get the file path
        path, _ = QFileDialog.getOpenFileName(
            self, 'Select a dose-influence matrix file', QDir.rootPath(),
            'Dose-influence matrix (*.mat *.npy)')

        # Check if the file path exists (user has not stopped the adding)
        if path:

            # Set the dose matrix path field to the path
            self.dose_path_ledit.setText(path)

    def add_initial_fluence_vector(self):
        """Add the initial fluence vector from a file."""

        # Get the file path
        path, _ = QFileDialog.getOpenFileName(
            self, 'Select an initial fluence vector file', QDir.rootPath(),
            'Fluence vector (*.json *.p *.txt)')

        # Check if the file path exists (user has not stopped the adding)
        if path:

            # Get the list of values from the file
            value_list = load_list_from_file(path)

            # Set the initial fluence vector field to the value list
            self.init_fluence_ledit.setText(str(value_list))

    def add_lower_var_bounds(self):
        """Add the lower variable bounds from a file."""

        # Get the file path
        path, _ = QFileDialog.getOpenFileName(
            self, 'Select a lower variable bounds file', QDir.rootPath(),
            'Fluence vector (*.json *.p *.txt)')

        # Check if the file path exists (user has not stopped the adding)
        if path:

            # Get the list of values from the file
            value_list = load_list_from_file(path)

            # Set the initial fluence vector field to the value list
            self.lower_var_ledit.setText(str(value_list))

    def add_upper_var_bounds(self):
        """Add the upper variable bounds from a file."""

        # Get the file path
        path, _ = QFileDialog.getOpenFileName(
            self, 'Select an upper variable bounds file', QDir.rootPath(),
            'Fluence vector (*.json *.p *.txt)')

        # Check if the file path exists (user has not stopped the adding)
        if path:

            # Get the list of values from the file
            value_list = load_list_from_file(path)

            # Set the initial fluence vector field to the value list
            self.upper_var_ledit.setText(str(value_list))

    def update_configuration(self):
        """Update the configuration parameters."""

        # Overwrite the configuration dictionary of the current instance
        self.plans[self.plan_ledit.text()].update(
            self.transform_configuration_to_dict())

        # 
        self.set_configuration()

        # 
        self.optimize_pbutton.setEnabled(False)
        self.evaluate_pbutton.setEnabled(False)
        self.visualize_pbutton.setEnabled(False)

    def update_optimization(self):
        """Update the optimization parameters."""

        # Overwrite the optimization dictionary of the current instance
        self.plans[self.plan_ledit.text()].update(
            self.transform_optimization_to_dict())

        # 
        self.set_optimization()

        # 
        self.evaluate_pbutton.setEnabled(False)
        self.visualize_pbutton.setEnabled(False)

    def update_evaluation(self):
        """Update the evaluation parameters."""

        # Overwrite the evaluation dictionary of the current instance
        self.plans[self.plan_ledit.text()].update(
            self.transform_evaluation_to_dict())

        # 
        self.set_evaluation()

    def clear_configuration(self):
        """Clear the configuration parameters."""

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

        # Clear the target imaging resolution
        self.img_res_ledit.clear()

        # Reset the dose matrix path
        self.dose_path_ledit.setText(
            self.base_configuration['dose_matrix_path'])

        # Reset the dose resolution
        self.dose_res_ledit.setText(
            self.base_configuration['dose_resolution'])

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

        # Reset the maximum CPU time
        self.max_cpu_ledit.setText(str(self.base_optimization['max_cpu_time']))

    def clear_evaluation(self):
        """Clear the evaluation parameters."""

        # Reset the DVH type
        self.dvh_type_cbox.setCurrentText(self.base_evaluation['dvh_type'])

        # Reset the number of DVH points
        self.n_points_sbox.setValue(self.base_evaluation['number_of_points'])

        # Reset the reference volume
        self.ref_vol_ledit.setText(
            str(self.base_evaluation['reference_volume']))

        # Reset the reference (fractional) dose values
        self.ref_dose_ledit.setText(
            str(self.base_evaluation['reference_dose']))

        # Reset the display segments
        self.display_segments_ledit.setText(
            str(self.base_evaluation['display_segments']))

        # Loop over the display metrics
        for index in range(self.display_metrics_lwidget.count()):

            # Reset the display metric to checked
            self.display_metrics_lwidget.item(index).setCheckState(2)

    def set_configuration(self):
        """Set the configuration parameters."""

        # Get the configuration dictionary from the current plan
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
        self.img_path_ledit.setText(configuration['imaging_path'])

        # Check if the target imaging resolution is not None
        if configuration['target_imaging_resolution']:

            # Set the target imaging resolution
            self.img_res_ledit.setText(
                str(configuration['target_imaging_resolution']))

        else:

            # Clear the target imaging resolution
            self.img_res_ledit.clear()

        # Set the dose matrix path
        self.dose_path_ledit.setText(
            configuration['dose_matrix_path'])

        # Set the dose resolution
        self.dose_res_ledit.setText(
            str(configuration['dose_resolution']))

    def set_optimization(self):
        """Set the optimization parameters."""

        # Get the optimization dictionary from the current plan
        optimization = self.plans[self.plan_ledit.text()].optimization

        # Clear the component list
        self.components_lwidget.clear()

        # Loop over the components
        for segment, component in optimization['components'].items():

            # Get the component type
            component_type = component['type']

            # Check if the component is an objective
            if component_type == 'objective':

                # Set the icon path to point towards a target symbol
                icon_path = (":/special_icons/icons_special/"
                             "target-red-svgrepo-com.svg")

            else:

                # Set the icon path to point towards a frame symbol
                icon_path = (":/special_icons/icons_special/"
                             "frame-red-svgrepo-com.svg")

            # Initialize the icon object
            icon = QIcon()

            # Add the pixmap to the icon
            icon.addPixmap(QPixmap(icon_path), QIcon.Normal, QIcon.Off)

            # Check if the component instance is not a list
            if isinstance(component['instance'], dict):

                # Get the component class name
                class_name = component['instance']['class']

                # Get the component identifier
                if 'identifier' in component['instance']['parameters']:
                    identifier = component['instance']['parameters'][
                        'identifier']
                else:
                    identifier = None

                # 
                if 'embedding' in component['instance']['parameters']:
                    embedding = ''.join(
                        ('embedding: ',
                         str(component['instance']['parameters']['embedding'])
                         ))
                else:
                    embedding = 'embedding: active'

                # 
                if 'weight' in component['instance']['parameters']:
                    weight = ''.join(
                        ('weight: ',
                         str(component['instance']['parameters']['weight'])))
                else:
                    weight = 'weight: 1'

                # Join the segment and class name
                component_string = ' - '.join((parameter for parameter in (
                    segment, class_name, identifier, embedding, weight)
                    if parameter))

                # Add the item to the list
                self.components_lwidget.addItem(
                    QListWidgetItem(icon, component_string))

            else:

                # Loop over the instances
                for instance in component['instance']:

                    # Get the component class name
                    class_name = instance['class']

                    # Get the component identifier
                    if 'identifier' in instance['parameters']:
                        identifier = instance['parameters']['identifier']
                    else:
                        identifier = None

                    # 
                    if 'embedding' in instance['parameters']:
                        embedding = ''.join(
                            ('embedding: ',
                             str(instance['parameters']['embedding'])
                             ))
                    else:
                        embedding = 'embedding: active'

                    # 
                    if 'weight' in instance['parameters']:
                        weight = ''.join(
                            ('weight: ',
                             str(instance['parameters']['weight'])))
                    else:
                        weight = 'weight: 1'

                    # Join the segment and class name
                    component_string = ' - '.join((parameter for parameter in (
                        segment, class_name, identifier, embedding, weight)
                        if parameter))

                    # Add the item to the list
                    self.components_lwidget.addItem(
                        QListWidgetItem(icon, component_string))

        # Set the optimization method
        self.method_cbox.setCurrentText(optimization['method'])

        # Set the solver package
        self.solver_cbox.setCurrentText(optimization['solver'])

        # Set the algorithm
        self.algorithm_cbox.setCurrentText(optimization['algorithm'])

        # Set the initialization strategy
        self.init_strat_cbox.setCurrentText(optimization['initial_strategy'])

        # 
        if self.init_strat_cbox.currentText() != 'warm-start':

            # 
            self.init_fluence_ledit.setEnabled(False)
            self.init_fluence_tbutton.setEnabled(False)
            self.ref_plan_cbox.setEnabled(False)

        # Check if the initial fluence vector is not None
        if optimization['initial_fluence_vector']:

            # Set the initial fluence vector
            self.init_fluence_ledit.setText(
                str(optimization['initial_fluence_vector']))

        else:

            # Clear the initial fluence vector
            self.init_fluence_ledit.clear()

        # Set the initial reference plan
        self.ref_plan_cbox.setCurrentIndex(0)

        # Check if the lower variable bounds is different from zero
        if optimization['lower_variable_bounds'] != 0:

            # Set the lower variable bounds
            self.lower_var_ledit.setText(
                str(optimization['lower_variable_bounds']))

        else:

            # Clear the lower variable bounds
            self.lower_var_ledit.clear()

        # Check if the upper variable bounds is not None
        if optimization['upper_variable_bounds']:

            # Set the upper variable bounds
            self.upper_var_ledit.setText(
                str(optimization['upper_variable_bounds']))

        else:

            # Clear the upper variable bounds
            self.upper_var_ledit.clear()

        # Set the maximum number of iterations
        self.max_iter_sbox.setValue(optimization['max_iter'])

        # Set the maximum CPU time
        self.max_cpu_ledit.setText(str(optimization['max_cpu_time']))

    def set_evaluation(self):
        """Set the evaluation parameters."""

        # Get the evaluation dictionary from the current plan
        evaluation = self.plans[self.plan_ledit.text()].evaluation

        # Set the DVH type
        self.dvh_type_cbox.setCurrentText(evaluation['dvh_type'])

        # Set the number of DVH points
        self.n_points_sbox.setValue(evaluation['number_of_points'])

        # Set the reference volume
        self.ref_vol_ledit.setText(str(evaluation['reference_volume']))

        # Set the reference dose values
        self.ref_dose_ledit.setText(str(evaluation['reference_dose']))

        # Set the display segments
        self.display_segments_ledit.setText(
            str(evaluation['display_segments']).replace("\'", ''))

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

    def transform_configuration_to_dict(self):
        """Transform the configuration fields into a dictionary."""

        # 
        target_imaging_resolution = make_list_string(
            self.img_res_ledit.text(), 0)
        dose_resolution = make_list_string(self.dose_res_ledit.text(), 0)

        # Create the configuration dictionary from the input fields
        configuration = {
            'label': None if not self.plan_ledit.text()
            else self.plan_ledit.text(),
            'min_log_level': self.log_level_cbox.currentText(),
            'modality': self.modality_cbox.currentText(),
            'number_of_fractions': self.nfx_sbox.value(),
            'imaging_path': None if not self.img_path_ledit.text()
            else self.img_path_ledit.text(),
            'target_imaging_resolution': None if not target_imaging_resolution
            else loads(target_imaging_resolution),
            'dose_matrix_path': None if not self.dose_path_ledit.text()
            else self.dose_path_ledit.text(),
            'dose_resolution': None if not dose_resolution
            else loads(dose_resolution)
            }

        return configuration

    def transform_optimization_to_dict(self):
        """Transform the optimization fields into a dictionary.."""

        # Initialize the initial fluence candidates
        candidates = [
            self.init_fluence_ledit.text(),
            self.ref_plan_cbox.currentText()]

        # 
        lower_variable_bounds = make_list_string(
            self.lower_var_ledit.text(), 1)
        upper_variable_bounds = make_list_string(
            self.upper_var_ledit.text(), 1)

        # Create the optimization dictionary from the input fields
        optimization = {
            'components': {
                'PAROTID_LT': {
                    'type': 'objective',
                    'instance': {
                        'class': 'Squared Overdosing',
                        'parameters': {
                            'maximum_dose': 25,
                            'weight': 100
                            }
                        }
                    },
                'PAROTID_RT': {
                    'type': 'objective',
                    'instance': {
                        'class': 'Squared Overdosing',
                        'parameters': {
                            'maximum_dose': 25,
                            'weight': 100
                            }
                        }
                    },
                'PTV63': {
                    'type': 'objective',
                    'instance': {
                        'class': 'Squared Deviation',
                        'parameters': {
                            'target_dose': 63,
                            'weight': 1000
                            }
                        }
                    },
                'PTV70': {
                    'type': 'objective',
                    'instance': {
                        'class': 'Squared Deviation',
                        'parameters': {
                            'target_dose': 70,
                            'weight': 1000
                            }
                        }
                    },
                'SKIN': {
                    'type': 'objective',
                    'instance': {
                        'class': 'Squared Overdosing',
                        'parameters': {
                            'maximum_dose': 30,
                            'weight': 800
                            }
                        }
                    }
                },
            'method': self.method_cbox.currentText(),
            'solver': self.solver_cbox.currentText(),
            'algorithm': self.algorithm_cbox.currentText(),
            'initial_strategy': self.init_strat_cbox.currentText(),
            'initial_fluence_vector': (None if candidates[0] == ''
                                       and candidates[1] == 'None' else
                                       loads(self.init_fluence_ledit.text())
                                       if candidates[0] != '' else
                                       self.plans[candidates[1]]
                                       .datahub.optimization[
                                           'optimized_fluence'].tolist()),
            'lower_variable_bounds': (0
                                      if not lower_variable_bounds
                                      else loads(lower_variable_bounds)
                                      ),
            'upper_variable_bounds': (None
                                      if not upper_variable_bounds
                                      else loads(upper_variable_bounds)
                                      ),
            'max_iter': self.max_iter_sbox.value(),
            'max_cpu_time': (3000.0
                             if self.max_cpu_ledit.text() == ''
                             else loads(
                                 self.max_cpu_ledit.text())
                             )
            }

        return optimization

    def transform_evaluation_to_dict(self):
        """Transform the evaluation fields into a dictionary.."""

        # 
        reference_volume = make_list_string(
            self.ref_vol_ledit.text(), 0)
        reference_dose = make_list_string(
            self.ref_dose_ledit.text(), 0)
        display_segments = self.display_segments_ledit.text()

        # Create the evaluation dictionary from the input fields
        evaluation = {
            'dvh_type': self.dvh_type_cbox.currentText(),
            'number_of_points': self.n_points_sbox.value(),
            'reference_volume': ([2, 5, 50, 95, 98]
                                 if reference_volume in ('', '[]')
                                 else loads(reference_volume)),
            'reference_dose': ([]
                               if reference_dose in ('', '[]')
                               else loads(reference_dose)),
            'display_segments': ([]
                                 if display_segments in ('', '[]')
                                 else display_segments.strip(
                                     '][').split(', ')),
            'display_metrics': [
                self.display_metrics_lwidget.item(index).text()
                for index in range(self.display_metrics_lwidget.count())
                if self.display_metrics_lwidget.item(index).checkState()]
            }

        return evaluation

    def update_by_plan_label(self):
        """."""

        # 
        if self.plan_ledit.text() in (
                '',
                *(self.plan_select_cbox.itemText(i)
                  for i in range(self.plan_select_cbox.count()))):

            # 
            self.update_configuration_pbutton.setEnabled(True)
            self.update_optimization_pbutton.setEnabled(True)
            self.update_evaluation_pbutton.setEnabled(True)

            # 
            self.reset_configuration_pbutton.setEnabled(True)
            self.reset_optimization_pbutton.setEnabled(True)
            self.reset_evaluation_pbutton.setEnabled(True)

            # 
            self.initialize_pbutton.setEnabled(False)

        else:

            # 
            self.update_configuration_pbutton.setEnabled(False)
            self.update_optimization_pbutton.setEnabled(False)
            self.update_evaluation_pbutton.setEnabled(False)

            # 
            self.reset_configuration_pbutton.setEnabled(False)
            self.reset_optimization_pbutton.setEnabled(False)
            self.reset_evaluation_pbutton.setEnabled(False)

            # 
            self.initialize_pbutton.setEnabled(True)

    def update_by_initial_strategy(self):
        """."""

        # 
        if self.init_strat_cbox.currentText() != 'warm-start':

            # 
            self.init_fluence_ledit.setEnabled(False)
            self.init_fluence_tbutton.setEnabled(False)
            self.ref_plan_cbox.setEnabled(False)

        else:

            # 
            self.init_fluence_ledit.setEnabled(True)
            self.init_fluence_tbutton.setEnabled(True)
            self.ref_plan_cbox.setEnabled(True)

    def update_by_reference(self):
        """."""

        # 
        if self.ref_plan_cbox.currentText() != 'None':

            # 
            self.init_fluence_ledit.setEnabled(False)
            self.init_fluence_tbutton.setEnabled(False)

        else:

            # 
            self.init_fluence_ledit.setEnabled(True)
            self.init_fluence_tbutton.setEnabled(True)

    def update_by_initial_fluence(self):
        """."""

        # 
        if self.init_fluence_ledit.text() != '':

            # 
            self.ref_plan_cbox.setEnabled(False)

        else:

            # 
            self.ref_plan_cbox.setEnabled(True)

    def update_by_method(self):
        """Update the solver options by the optimization method."""

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
            self.solver_cbox.addItems(['ipopt', 'proxmin', 'pypop7', 'scipy'])

            # Set the default solver option
            self.solver_cbox.setCurrentText('scipy')

    def update_by_solver(self):
        """Update the solution algorithm by the solver."""

        # Clear the algorithm combo box
        self.algorithm_cbox.clear()

        # Check if the solver is 'ipopt'
        if self.solver_cbox.currentText() == 'ipopt':

            # Add the algorithm options to the combo box
            self.algorithm_cbox.addItems(['ma27', 'ma57', 'ma77', 'ma86'])

            # Set the default algorithm option
            self.algorithm_cbox.setCurrentText('ma57')

        # Else, check if the solver is 'proxmin'
        elif self.solver_cbox.currentText() == 'proxmin':

            # Add the algorithm options to the combo box
            self.algorithm_cbox.addItems(['admm', 'pgm', 'sdmm'])

            # Set the default algorithm option
            self.algorithm_cbox.setCurrentText('pgm')

        # Else, check if the solver is 'pymoo'
        elif self.solver_cbox.currentText() == 'pymoo':

            # Add the algorithm options to the combo box
            self.algorithm_cbox.addItems(['NSGA-3'])

            # Set the default algorithm option
            self.algorithm_cbox.setCurrentText('NSGA-3')

        # Else, check if the solver is 'pypop7'
        elif self.solver_cbox.currentText() == 'pypop7':

            # Add the algorithm options to the combo box
            self.algorithm_cbox.addItems(
                ['MMES', 'LMCMA', 'RMES', 'BES', 'GS'])

            # Set the default algorithm option
            self.algorithm_cbox.setCurrentText('LMCMA')

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

    def open_drop_tpi_dialog(self):
        """."""

        # 
        if self.plan_select_cbox.currentText() != '':

            # 
            message = ("Dropping the current treatment plan will irreversibly "
                       "remove it from the GUI. Are you sure you want to "
                       "proceed?")

            # 
            if (QMessageBox.question(self, "pyanno4rt ", message)
                    == QMessageBox.Yes):

                # 
                self.drop_tpi()

            else:

                pass

    def open_update_configuration_dialog(self):
        """."""

        # 
        message = ("Updating will change the configuration parameters of the "
                   "current treatment plan. Are you sure you want to proceed?")

        # 
        if (QMessageBox.question(self, "pyanno4rt ", message)
                == QMessageBox.Yes):

            # 
            self.update_configuration()

        else:

            pass

    def open_update_optimization_dialog(self):
        """."""

        # 
        message = ("Updating will change the optimization parameters of the "
                   "current treatment plan. Are you sure you want to proceed?")

        # 
        if (QMessageBox.question(self, "pyanno4rt ", message)
                == QMessageBox.Yes):

            # 
            self.update_optimization()

        else:

            pass

    def open_update_evaluation_dialog(self):
        """."""

        # 
        message = ("Updating will change the evaluation parameters of the "
                   "current treatment plan. Are you sure you want to proceed?")

        # 
        if (QMessageBox.question(self, "pyanno4rt ", message)
                == QMessageBox.Yes):

            # 
            self.update_evaluation()

        else:

            pass

    def open_reset_configuration_dialog(self):
        """."""

        # 
        message = ("Resetting will change all configuration tab fields to the "
                   "last saved state. Are you sure you want to proceed?")

        # 
        if (QMessageBox.question(self, "pyanno4rt ", message)
                == QMessageBox.Yes):

            # 
            self.set_configuration()

        else:

            pass

    def open_reset_optimization_dialog(self):
        """."""

        # 
        message = ("Resetting will change all optimization tab fields to the "
                   "last saved state. Are you sure you want to proceed?")

        # 
        if (QMessageBox.question(self, "pyanno4rt ", message)
                == QMessageBox.Yes):

            # 
            self.set_optimization()

        else:

            pass

    def open_reset_evaluation_dialog(self):
        """."""

        # 
        message = ("Resetting will change all evaluation tab fields to the "
                   "last saved state. Are you sure you want to proceed?")

        # 
        if (QMessageBox.question(self, "pyanno4rt ", message)
                == QMessageBox.Yes):

            # 
            self.set_evaluation()

        else:

            pass

    def open_clear_configuration_dialog(self):
        """."""

        # 
        message = ("Clearing will change all configuration tab fields to the "
                   "default state. Are you sure you want to proceed?")

        # 
        if (QMessageBox.question(self, "pyanno4rt ", message)
                == QMessageBox.Yes):

            # 
            self.clear_configuration()

        else:

            pass

    def open_clear_optimization_dialog(self):
        """."""

        # 
        message = ("Clearing will change all optimization tab fields to the "
                   "default state. Are you sure you want to proceed?")

        # 
        if (QMessageBox.question(self, "pyanno4rt ", message)
                == QMessageBox.Yes):

            # 
            self.clear_optimization()

        else:

            pass

    def open_clear_evaluation_dialog(self):
        """."""

        # 
        message = ("Clearing will change all evaluation tab fields to the "
                   "default state. Are you sure you want to proceed?")

        # 
        if (QMessageBox.question(self, "pyanno4rt ", message)
                == QMessageBox.Yes):

            # 
            self.clear_evaluation()

        else:

            pass