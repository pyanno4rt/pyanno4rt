"""Graphical user interface."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from os.path import dirname
from itertools import islice, cycle
from json import loads
from json import load as jload
from matplotlib import colormaps
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from numpy import (nan, ndarray, rot90, transpose, unravel_index, zeros)
from pickle import load
from PyQt5.QtCore import QDir, Qt
from PyQt5.QtGui import QCursor, QIcon, QPixmap
from PyQt5.QtWidgets import (QApplication, QFileDialog, QListWidgetItem,
                             QMainWindow, QMessageBox, QVBoxLayout, QWidget)
from pyqtgraph import (colormap, GraphicsLayoutWidget, ImageItem, InfiniteLine,
                       IsocurveItem, mkQApp, mkColor, mkPen, PlotWidget,
                       setConfigOptions, SignalProxy)

# %% Internal package import

from pyanno4rt.base import TreatmentPlan
from pyanno4rt.gui.assets import resources_rc
from pyanno4rt.gui.windows.main_window import Ui_main_window
from pyanno4rt.gui.windows.settings_window import Ui_settings_window
from pyanno4rt.gui.windows.info_window import Ui_info_window
from pyanno4rt.tools import copycat, snapshot

# %% Set options

setConfigOptions(imageAxisOrder='col-major')

# %% Class definition


class GraphicalUserInterface():
    """
    Graphical user interface class.

    This class provides ...

    Parameters
    ----------
    ...

    Attributes
    ----------
    ...
    """

    def __init__(self):

        # Initialize the application
        self.application = mkQApp("Graphical User Interface")

        # Set the application style
        self.application.setStyle('Fusion')

    def launch(
            self,
            treatment_plan=None):
        """Launch the graphical user interface."""

        # Initialize the main window for the GUI
        self.main_window = MainWindow(treatment_plan, self.application)

        # Run the application
        self.main_window.application.exec_()

    def get_plans(self):
        """Get the treatment plan dictionary of the GUI."""

        return self.main_window.plans

    def closeEvent(
            self,
            event):
        """
        Close the application.

        Parameters
        ----------
        event : object of class `QCloseEvent`
            Instance of the class `QCloseEvent` to be triggered at window \
            closing.
        """

        # Close the application
        self.application.quit()


class MainWindow(QMainWindow, Ui_main_window):
    """
    Main window for the application.

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

        # Initialize the GUI plans dictionary
        self.plans = {}

        # Initialize the optimized plans list
        self.optimized_plans = []

        # Get the base dictionaries
        self.base_configuration = self.transform_configuration_to_dict()
        self.base_optimization = self.transform_optimization_to_dict()
        self.base_evaluation = self.transform_evaluation_to_dict()

        # Disable the update buttons initially
        self.update_configuration_pbutton.setEnabled(False)
        self.update_optimization_pbutton.setEnabled(False)
        self.update_evaluation_pbutton.setEnabled(False)

        # Disable the workflow buttons
        self.optimize_pbutton.setEnabled(False)
        self.evaluate_pbutton.setEnabled(False)
        self.visualize_pbutton.setEnabled(False)

        # Initialize the slice widget
        self.slice_widget = SliceWidget(self)

        # Insert the slice widget into the viewer
        self.tab_slices_layout.insertWidget(0, self.slice_widget)

        # Initialize the DVH widget
        self.dvh_widget = DVHWidget(parent=self)

        # Insert the slice widget into the viewer
        self.tab_dvh_layout.insertWidget(0, self.dvh_widget)

        # Initialize the settings window
        self.settings_window = SettingsWindow(self)

        # Initialize the information window
        self.info_window = InfoWindow(self)

        # Set the scrollbar stylesheet for the console output
        self.console_tedit.verticalScrollBar().setStyleSheet('''
                                  QScrollBar
                                      {
                                          color: black;
                                          background-color: #d3d7cf;
                                      }
                                 QScrollBar::sub-page
                                     {
                                         background: black;
                                         border: 1px solid #d3d7cf;
                                     }
                                 QScrollBar::add-page
                                     {
                                         background: black;
                                         border: 1px solid #d3d7cf;
                                     }
                                 ''')

        # Set the tab indices to ensure consistency of the starting window
        self.composer_widget.setCurrentIndex(0)
        self.tab_workflow.setCurrentIndex(0)
        self.viewer_widget.setCurrentIndex(0)

        # Check if an initial treatment plan has been specified
        if treatment_plan:

            # Activate the treatment plan in the GUI
            self.activate(treatment_plan)

        # Connect the event signals
        self.connect_signals()

        # Show the GUI
        self.showMaximized()

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
        self.settings_pbutton.clicked.connect(self.open_settings)
        self.info_pbutton.clicked.connect(self.open_info)
        self.exit_pbutton.clicked.connect(self.exit_window)

        # Connect the workflow buttons
        self.configure_pbutton.clicked.connect(self.configure)
        self.optimize_pbutton.clicked.connect(self.optimize)
        self.evaluate_pbutton.clicked.connect(self.evaluate)
        self.visualize_pbutton.clicked.connect(self.visualize)
        self.compose_pbutton.clicked.connect(self.compose)

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

        # Connect the optimization field dependencies
        self.method_cbox.currentIndexChanged.connect(
            self.update_solver_by_method)
        self.solver_cbox.currentIndexChanged.connect(
            self.update_algorithm_by_solver)

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

        # Connect the viewer elements
        self.opacity_sbox.valueChanged.connect(self.change_dose_opacity)
        self.slice_selection_sbar.valueChanged.connect(self.change_image_slice)

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

        # Check if the selector text is not set to default
        if self.plan_select_cbox.currentText() != '':

            # Change the treatment plan label to the instance label
            self.plan_ledit.setText(self.plan_select_cbox.currentText())

            # Set the configuration, optimization and evaluation tab fields
            self.set_configuration()
            self.set_optimization()
            self.set_evaluation()

            # Enable the update buttons
            self.update_configuration_pbutton.setEnabled(True)
            self.update_optimization_pbutton.setEnabled(True)
            self.update_evaluation_pbutton.setEnabled(True)

        else:

            # Change the treatment plan label to the default (empty)
            self.plan_ledit.setText('')

            # Reset the configuration, optimization and evaluation tab fields
            self.reset_configuration()
            self.reset_optimization()
            self.reset_evaluation()

            # Disable the update buttons
            self.update_configuration_pbutton.setEnabled(False)
            self.update_optimization_pbutton.setEnabled(False)
            self.update_evaluation_pbutton.setEnabled(False)

        # Disable the workflow buttons
        self.optimize_pbutton.setEnabled(False)
        self.evaluate_pbutton.setEnabled(False)
        self.visualize_pbutton.setEnabled(False)

        # Clear the console text
        self.console_tedit.clear()

        # Reset the slice image
        self.slice_widget.reset_images()

        # Reset the DVH plot
        self.dvh_widget.reset_dvh()

    def open_settings(self):
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

    def open_info(self):
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

    def configure(self):
        """Configure the treatment plan."""

        # Set the waiting cursor
        QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))

        # Try to configure the plan
        try:

            # Initialize the treatment plan instance
            self.initialize_base_class()

            # Run the configure method of the instance
            self.plans[self.plan_ledit.text()].configure()

            # Check if the instance has not been deposited in the selector
            if self.plan_ledit.text() not in (
                    self.plan_select_cbox.itemText(i)
                    for i in range(self.plan_select_cbox.count())):

                # Add the instance to the selector
                self.plan_select_cbox.addItem(self.plan_ledit.text())

            # Change the selector text to the configured instance
            self.plan_select_cbox.setCurrentText(self.plan_ledit.text())

            # Update the console text for the new instance
            self.update_console_output()

            # Get the CT dictionary
            computed_tomography = (self.plans[self.plan_ledit.text()]
                                   .datahub.computed_tomography)

            # Get the segmentation dictionary
            segmentation = (self.plans[self.plan_ledit.text()]
                            .datahub.segmentation)

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

            # Update the console text
            self.update_console_output()

            # Get the optimized dose array
            optimized_dose = instance.datahub.optimization['optimized_dose']

            # Add the dose image to the widget
            self.slice_widget.add_dose(optimized_dose)

            # Update the slice image
            self.slice_widget.update_images()

            # Enable the evaluation button
            self.evaluate_pbutton.setEnabled(True)

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

            # Run the evaluate method of the instance
            self.plans[self.plan_ledit.text()].evaluate()

            # Update the console text
            self.update_console_output()

            # 
            self.dvh_widget.reset_dvh()

            # 
            histogram = (self.plans[self.plan_ledit.text()]
                         .datahub.histogram)

            # 
            self.dvh_widget.add_style_and_data(histogram)

            # 
            self.dvh_widget.update_dvh()

            # Enable the visualization button
            self.visualize_pbutton.setEnabled(True)

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

            # Update the console text
            self.update_console_output()

        except Exception as error:

            # 
            QMessageBox.warning(self, "pyanno4rt", str(error))

    def compose(self):
        """Compose the treatment plan."""

        # Run the configure, optimize, evaluate and visualize methods of the
        # instance (this represents a modified compose method)
        is_valid = self.configure()

        # 
        if is_valid:

            self.optimize()
            self.evaluate()
            self.visualize()

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
            value_list = self.load_list_from_file(path)

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
            value_list = self.load_list_from_file(path)

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
            value_list = self.load_list_from_file(path)

            # Set the initial fluence vector field to the value list
            self.upper_var_ledit.setText(str(value_list))

    def update_configuration(self):
        """Update the configuration parameters."""

        # Overwrite the configuration dictionary of the current instance
        self.plans[self.plan_ledit.text()].update(
            self.transform_configuration_to_dict())

        # 
        self.set_configuration()

    def update_optimization(self):
        """Update the optimization parameters."""

        # Overwrite the optimization dictionary of the current instance
        self.plans[self.plan_ledit.text()].update(
            self.transform_optimization_to_dict())

        # 
        self.set_optimization()

    def update_evaluation(self):
        """Update the evaluation parameters."""

        # Overwrite the evaluation dictionary of the current instance
        self.plans[self.plan_ledit.text()].update(
            self.transform_evaluation_to_dict())

        # 
        self.set_evaluation()

    def reset_configuration(self):
        """Reset the configuration parameters."""

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

    def reset_optimization(self):
        """Reset the optimization parameters."""

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

    def reset_evaluation(self):
        """Reset the evaluation parameters."""

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
            component_type = component[0]

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

            # Get the component class name
            class_name = component[1]['class']

            # Get the component identifier
            if 'identifier' in component[1]['parameters']:
                identifier = component[1]['parameters']['identifier']
            else:
                identifier = None

            # 
            if 'embedding' in component[1]['parameters']:
                embedding = ''.join(
                    ('embedding: ',
                     str(component[1]['parameters']['embedding'])))
            else:
                embedding = 'embedding: active'

            # 
            if 'weight' in component[1]['parameters']:
                weight = ''.join(
                    ('weight: ', str(component[1]['parameters']['weight'])))
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
            str(evaluation['display_segments']))

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

    def load_list_from_file(
            self,
            path):
        """Load a list of values from a file."""

        # Check if a JSON file has been selected
        if path.endswith('.json'):

            # Open a file stream
            with open(path, 'rb') as file:

                # Get the list of values
                value_list = jload(file)

        # Check if a Python file has been selected
        elif path.endswith('.p'):

            # Open a file stream
            with open(path, 'rb') as file:

                # Get the list of values
                value_list = load(file)

        # Check if a text file has been selected
        elif path.endswith('.txt'):

            # Open a file stream
            with open(path, 'r') as file:

                # Get the list of values
                value_list = [float(line.rstrip('\n')) for line in file]

        return value_list

    def make_list_string(self, text, min_length):
        """."""

        # 
        text.replace('(', '[')
        text.replace(')', ']')

        # 
        if len(text) > min_length and (text[0], text[-1]) != ('[', ']'):

            # 
            return ''.join(('[', text, ']'))

        else:

            # 
            return text

    def initialize_base_class(self):
        """Initialize the base treatment plan class."""

        # Check if the treatment plan is not yet a validated instance
        if self.plan_ledit.text() not in (*self.plans,):

            # Initialize the base class and add the instance to the dictionary
            self.plans[self.plan_ledit.text()] = TreatmentPlan(
                self.transform_configuration_to_dict(),
                self.transform_optimization_to_dict(),
                self.transform_evaluation_to_dict())

    def transform_configuration_to_dict(self):
        """Transform the configuration fields into a dictionary."""

        # 
        target_imaging_resolution = self.make_list_string(
            self.img_res_ledit.text(), 0)
        dose_resolution = self.make_list_string(self.dose_res_ledit.text(), 0)

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
        lower_variable_bounds = self.make_list_string(
            self.lower_var_ledit.text(), 1)
        upper_variable_bounds = self.make_list_string(
            self.upper_var_ledit.text(), 1)

        # Create the optimization dictionary from the input fields
        optimization = {
            'components': {
                'PAROTID_LT': ['objective',
                               {'class': 'Squared Overdosing',
                                'parameters': {'maximum_dose': 25,
                                               'weight': 100}}
                               ],
                'PAROTID_RT': ['objective',
                               {'class': 'Squared Overdosing',
                                'parameters': {'maximum_dose': 25,
                                               'weight': 100}}
                               ],
                'PTV63': ['objective',
                          {'class': 'Squared Deviation',
                           'parameters': {'target_dose': 63,
                                          'weight': 1000}}
                          ],
                'PTV70': ['objective',
                          {'class': 'Squared Deviation',
                           'parameters': {'target_dose': 70,
                                          'weight': 1000}}
                          ],
                'SKIN': ['objective',
                         {'class': 'Squared Overdosing',
                          'parameters': {'maximum_dose': 30,
                                         'weight': 800}}
                         ],
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
        reference_volume = self.make_list_string(
            self.ref_vol_ledit.text(), 0)
        reference_dose = self.make_list_string(
            self.ref_dose_ledit.text(), 0)
        display_segments = self.make_list_string(
            self.display_segments_ledit.text(), 0)

        # Create the evaluation dictionary from the input fields
        evaluation = {
            'dvh_type': self.dvh_type_cbox.currentText(),
            'number_of_points': self.n_points_sbox.value(),
            'reference_volume': ([2, 5, 50, 95, 98]
                                 if reference_volume == ''
                                 else loads(reference_volume)),
            'reference_dose': [] if reference_dose == ''
            else loads(reference_dose),
            'display_segments': [] if display_segments == ''
            else loads(display_segments),
            'display_metrics': [
                self.display_metrics_lwidget.item(index).text()
                for index in range(self.display_metrics_lwidget.count())
                if self.display_metrics_lwidget.item(index).checkState()]
            }

        return evaluation

    def update_solver_by_method(self):
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

    def update_algorithm_by_solver(self):
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
                   "default values. Are you sure you want to proceed?")

        # 
        if (QMessageBox.question(self, "pyanno4rt ", message)
                == QMessageBox.Yes):

            # 
            self.reset_configuration()

        else:

            pass

    def open_reset_optimization_dialog(self):
        """."""

        # 
        message = ("Resetting will change all optimization tab fields to the "
                   "default values. Are you sure you want to proceed?")

        # 
        if (QMessageBox.question(self, "pyanno4rt ", message)
                == QMessageBox.Yes):

            # 
            self.reset_optimization()

        else:

            pass

    def open_reset_evaluation_dialog(self):
        """."""

        # 
        message = ("Resetting will change all evaluation tab fields to the "
                   "default values. Are you sure you want to proceed?")

        # 
        if (QMessageBox.question(self, "pyanno4rt ", message)
                == QMessageBox.Yes):

            # 
            self.reset_evaluation()

        else:

            pass

    def update_console_output(self):
        """."""

        # 
        self.console_tedit.clear()

        # 
        instance = self.plans[self.plan_ledit.text()]

        # 
        stream_value = instance.logger.logger.handlers[1].stream.getvalue()

        # 
        stream_value = stream_value.replace('\n', '\n\n')

        # 
        self.console_tedit.setText(stream_value)

    def change_dose_opacity(self):
        """."""

        # 
        self.slice_widget.dose_image.setOpacity(self.opacity_sbox.value()/100)

        # 
        self.slice_widget.update_images()

    def change_image_slice(self):
        """."""

        # 
        self.slice_widget.slice = self.slice_selection_sbar.value()

        # 
        self.slice_widget.update_images()

    def get_segment_statistics(self, event):
        """."""

        # 
        dosimetrics = (self.plans[self.plan_ledit.text()]
                       .datahub.dosimetrics)

        # 
        self.segment_ledit.setText(event.name())

        # 
        self.mean_ledit.setText(str(
            round(dosimetrics[event.name()]['mean'], 2)))

        # 
        self.std_ledit.setText(str(
            round(dosimetrics[event.name()]['std'], 2)))

        # 
        self.maximum_ledit.setText(str(
            round(dosimetrics[event.name()]['max'], 2)))

        # 
        self.minimum_ledit.setText(str(
            round(dosimetrics[event.name()]['min'], 2)))

    def select_dvh_curve(self, event):
        """."""

        # Get all plot items
        items = self.dvh_widget.plot_graph.getPlotItem().listDataItems()

        for item in items:
            pen = item.curve.opts['pen']
            if item == event:
                pen = mkPen(pen)
                if pen.width() == 2:
                    event.curve.setPen(mkPen(color=pen.color(),
                                             style=pen.style(),
                                             width=5))
                else:
                    event.curve.setPen(mkPen(color=pen.color(),
                                             style=pen.style(),
                                             width=2))
            else:
                item.curve.setPen(mkPen(color=pen.color(),
                                        style=pen.style(),
                                        width=2))

    def update_crosshair(
            self,
            event):
        """Update the crosshair at mouse moves."""

        # Get the coordinates from the triggered event
        coordinates = event[0]

        # Get the graph
        graph = self.dvh_widget.plot_graph

        # Check if the coordinates lie within the scene bounding rectangle
        if graph.sceneBoundingRect().contains(coordinates):

            # Get the mouse point in the view's coordinate system
            mouse_point = graph.plotItem.vb.mapSceneToView(coordinates)

            if 0 <= mouse_point.x() and 0 <= mouse_point.y() <= 100:

                # Update the graph title
                self.dvh_widget.plot_graph.setTitle(
                    "<span style='color: #FFAE42; "
                    "font-size: 11pt'>dose/fx: "
                    "%0.1f</span>, <span style='color: #FFAE42; "
                    "font-size: 11pt'>vRel: %0.1f</span>"
                    % (mouse_point.x(), mouse_point.y()))

            # Update the positions of vertical and horizontal lines
            self.dvh_widget.vertical_line.setPos(mouse_point.x())
            self.dvh_widget.horizontal_line.setPos(mouse_point.y())


class SliceWidget(QWidget):
    """."""

    def __init__(self, parent=None):

        # Call the superclass constructor
        super().__init__()

        # Set the vertical layout for the slice widget
        slice_layout = QVBoxLayout(self)

        # Create an image window, set its size, and add it to the slice layout
        image_window = GraphicsLayoutWidget()
        slice_layout.addWidget(image_window)

        # Add the view box to the image window
        self.viewbox = image_window.addViewBox()

        # 
        self.ct_image = ImageItem()
        self.viewbox.addItem(self.ct_image)

        # 
        self.dose_image = ImageItem()
        self.dose_image.setOpacity(0.7)
        colormap_dose = colormaps['jet']
        colormap_dose._init()
        self.dose_image.setLookupTable((colormap_dose._lut*255).view(ndarray))
        self.viewbox.addItem(self.dose_image)

        # 
        self.slice = None

        # 
        self.ct_cube = None
        self.dose_cube = None
        self.dose_cube_with_nan = None
        self.dose_contours = None
        self.segment_masks = None
        self.segment_contours = None

    def add_ct(self, ct_cube):
        self.ct_cube = rot90(transpose(ct_cube, (0, 1, 2)), 3)

    def add_dose(self, dose_cube):

        self.dose_cube = rot90(transpose(dose_cube, (0, 1, 2)), 3)

        self.dose_cube_with_nan = self.dose_cube.copy()
        self.dose_cube_with_nan[self.dose_cube_with_nan == 0] = nan

        quantiles = [0.1*factor1 for factor1 in range(1, 10)]
        quantiles.extend([0.95+0.05*factor2 for factor2 in range(0, 6)])

        reference_dose = self.dose_cube.max()/1.2

        levels = [reference_dose*level for level in quantiles]
        norm = Normalize(vmin=min(levels), vmax=max(levels), clip=True)
        mapper = ScalarMappable(norm=norm, cmap=colormaps['jet'])

        self.dose_contours = []
        for level in levels:
            contour = IsocurveItem(level=level, pen=mkPen(
                tuple([255*rgba for rgba in mapper.to_rgba(level)]),
                width=2.5))
            contour.setParentItem(self.dose_image)
            contour.setZValue(5)
            self.dose_contours.append(contour)

    def add_segments(self, computed_tomography, segmentation):

        def generate_segment_mask(segment):
            """Generate the segmentation masks as a single cube."""
            # Initialize the segment mask
            segment_mask = zeros(computed_tomography['cube_dimensions'])

            # Insert ones at the indices of the segment
            segment_mask[unravel_index(
                segmentation[segment]['raw_indices'],
                computed_tomography['cube_dimensions'], order='F')] = 1

            return segment_mask

        segment_colors = tuple(
            255*segmentation[segment]['parameters']['visibleColor']
            for segment in (*segmentation,))

        raw_masks = tuple(
            generate_segment_mask(segment) for segment in (*segmentation,))

        segment_images = [ImageItem() for _ in raw_masks]
        for image in segment_images:
            self.viewbox.addItem(image)

        self.segment_masks = tuple(rot90(
            transpose(mask, (0, 1, 2)), 3) for mask in raw_masks)

        self.segment_contours = []
        for color, image in zip(segment_colors, segment_images):
            contour = IsocurveItem(level=1, pen=mkPen(mkColor(color),
                                                      width=2.5))
            contour.setParentItem(image)
            contour.setZValue(5)
            self.segment_contours.append(contour)

    def update_images(self):
        """Update the images when scrolling."""

        if self.ct_cube is not None:
            # Update the CT image
            self.ct_image.setImage(self.ct_cube[:, :, self.slice])

        if self.dose_cube_with_nan is not None:
            # Update the dose image
            self.dose_image.setImage(self.dose_cube_with_nan[:, :, self.slice])

        if self.dose_cube is not None and self.dose_contours is not None:
            # Loop over the dose contours
            for contour in self.dose_contours:

                # Update the dose contour lines
                contour.setData(self.dose_cube[:, :, self.slice])

        if (self.segment_masks is not None
                and self.segment_contours is not None):
            # Loop over the segment contours
            for mask, contour in zip(self.segment_masks,
                                     self.segment_contours):

                # Update the segment contour lines
                contour.setData(mask[:, :, self.slice])

    def reset_images(self):
        """."""

        if self.ct_cube is not None:
            # Update the CT image
            self.ct_image.clear()
            self.ct_cube = None

        if self.dose_cube_with_nan is not None:
            # Update the dose image
            self.dose_image.clear()
            self.dose_cube_with_nan = None

        if self.dose_cube is not None or self.dose_contours is not None:
            # Loop over the dose contours
            for contour in self.dose_contours:

                # Update the dose contour lines
                contour.setData(zeros(self.dose_cube[:, :, self.slice].shape))

            self.dose_cube = None
            self.dose_contours = None

        if (self.segment_masks is not None
                or self.segment_contours is not None):
            # Loop over the segment contours
            for mask, contour in zip(self.segment_masks,
                                     self.segment_contours):

                # Update the segment contour lines
                contour.setData(zeros(mask[:, :, self.slice].shape))

            self.segment_masks = None
            self.segment_contours = None


class DVHWidget(QWidget):
    """."""

    def __init__(self, parent=None):

        # Call the superclass constructor
        super().__init__()

        self.parent = parent

        # Set the vertical layout for the DVH widget
        dvh_layout = QVBoxLayout(self)

        # 
        self.plot_graph = PlotWidget()

        # 
        dvh_layout.addWidget(self.plot_graph)

        self.plot_graph.getPlotItem().hideAxis('bottom')
        self.plot_graph.getPlotItem().hideAxis('left')
        self.plot_graph.getPlotItem().hideButtons()
        self.plot_graph.getPlotItem().setMouseEnabled(y=False)

        self.segments = None
        self.histogram = None
        self.segment_styles = None

        # Create vertical and horizontal infinite lines
        self.vertical_line = InfiniteLine(angle=90)
        self.horizontal_line = InfiniteLine(angle=0, movable=False)

        # Disable the pens for the lines
        self.vertical_line.setPen(None)
        self.horizontal_line.setPen(None)

        # Add the lines to the graph
        self.plot_graph.addItem(self.vertical_line, ignoreBounds=True)
        self.plot_graph.addItem(self.horizontal_line, ignoreBounds=True)

        # Set the signal proxy to update the crosshair at mouse moves
        self.crosshair_update = SignalProxy(
            self.plot_graph.scene().sigMouseMoved, rateLimit=60,
            slot=self.parent.update_crosshair)

    def add_style_and_data(self, histogram):
        """."""

        # 
        self.histogram = histogram

        # 
        self.segments = tuple(segment for segment in (*histogram,)
                              if segment not in ('evaluation_points',
                                                 'display_segments'))

        # Get the colormap
        colors = colormap.get('turbo', 'matplotlib').getLookupTable(
            nPts=len(self.segments))

        # Set the line styles
        line_styles = tuple(islice(cycle([Qt.SolidLine, Qt.DashLine,
                                          Qt.DotLine, Qt.DashDotLine]),
                                   len(self.segments)))

        # Create a dictionary with the segment styles
        self.segment_styles = dict(
            zip(self.segments, tuple(zip(colors, line_styles))))

        # 
        maximum_x = max(histogram['evaluation_points'])

        self.plot_graph.getPlotItem().showAxis('bottom')
        self.plot_graph.getPlotItem().showAxis('left')

        self.plot_graph.showGrid(x=True, y=True, alpha=0.3)
        self.plot_graph.setLabels(left="Relative volume [%]",
                                  bottom="Dose per fraction [Gy]")
        self.plot_graph.setXRange(0, maximum_x)
        self.plot_graph.setYRange(0, 105)

        # Set the graph title
        self.plot_graph.setTitle("<span style='color: #FFAE42; "
                                 "font-size: 11pt'>dose/fx: %0.1f</span>, "
                                 "<span style='color: #FFAE42; "
                                 "font-size: 11pt'>vRel: %0.1f</span>"
                                 % (0, 0.0))

        self.plot_graph.plotItem.vb.setLimits(xMin=0, xMax=maximum_x,
                                              yMin=0, yMax=101)

    def update_dvh(self):
        """."""

        for segment in self.segments:

            pen = mkPen(color=self.segment_styles[segment][0],
                        style=self.segment_styles[segment][1],
                        width=2)

            plot = self.plot_graph.plot(self.histogram['evaluation_points'],
                                        self.histogram[segment]['dvh_values'],
                                        pen=pen, name=segment, clickable=True)
            plot.sigClicked.connect(self.parent.get_segment_statistics)
            plot.sigClicked.connect(self.parent.select_dvh_curve)

    def reset_dvh(self):
        """."""

        self.plot_graph.clear()
        self.plot_graph.getPlotItem().hideAxis('bottom')
        self.plot_graph.getPlotItem().hideAxis('left')
        self.parent.segment_ledit.clear()
        self.parent.mean_ledit.clear()
        self.parent.std_ledit.clear()
        self.parent.maximum_ledit.clear()
        self.parent.minimum_ledit.clear()


class SettingsWindow(QMainWindow, Ui_settings_window):
    """
    Settings window for the application.

    This class creates the settings window for the graphical user interface, \
    including some user-definable parameters.
    """

    def __init__(
            self,
            parent=None):

        # Get the application from the argument
        self.parent = parent

        # Run the constructor from the superclass
        super().__init__()

        # Build the UI main window
        self.setupUi(self)

        # 
        self.default = ('English', 'Dark', (1920, 1080), (False, False, False))
        self.current = None

        # Temporarily disable combo boxes
        self.language_cbox.setEnabled(False)
        self.light_mode_cbox.setEnabled(False)

        # 
        self.reset_pbutton.clicked.connect(self.reset)
        self.save_pbutton.clicked.connect(self.save_apply_close)

    def position(self):
        """."""

        # Get the window geometry
        geometry = self.geometry()

        # Move the geometry center towards the parent
        geometry.moveCenter(self.parent.geometry().center())

        # Set the shifted geometry
        self.setGeometry(geometry)

    def get_fields(self):
        """."""

        # 
        language = self.language_cbox.currentText()

        # 
        light_mode = self.light_mode_cbox.currentText()

        # 
        resolution = tuple(
            map(loads, self.resolution_cbox.currentText().split('x')))

        # 
        includes = (self.incl_patData_check.isChecked(),
                    self.incl_dij_check.isChecked(),
                    self.incl_model_data_check.isChecked())

        return (language, light_mode, resolution, includes)

    def set_fields(self, settings):
        """."""

        # 
        self.language_cbox.setCurrentText(settings[0])

        # 
        self.light_mode_cbox.setCurrentText(settings[1])

        # 
        self.resolution_cbox.setCurrentIndex(0)
        self.resolution_cbox.setCurrentText('x'.join(map(str, settings[2])))

        # 
        self.incl_patData_check.setCheckState(2*settings[3][0])
        self.incl_dij_check.setCheckState(2*settings[3][1])
        self.incl_model_data_check.setCheckState(2*settings[3][2])

    def reset(self):
        """."""

        self.set_fields(self.default)

    def save_apply_close(self):
        """."""

        # 
        self.current = self.get_fields()

        # 
        self.parent.resize(*self.current[2])

        # 
        self.hide()


class InfoWindow(QMainWindow, Ui_info_window):
    """
    Information window for the application.

    This class creates the information window for the graphical user \
    interface, including some user-definable parameters.
    """

    def __init__(
            self,
            parent=None):

        # Get the application from the argument
        self.parent = parent

        # Run the constructor from the superclass
        super().__init__()

        # Build the UI main window
        self.setupUi(self)

        # 
        self.close_pbutton.clicked.connect(self.close)

    def position(self):
        """."""

        # Get the window geometry
        geometry = self.geometry()

        # Move the geometry center towards the parent
        geometry.moveCenter(self.parent.geometry().center())

        # Set the shifted geometry
        self.setGeometry(geometry)

    def close(self):
        """."""

        # 
        self.hide()
