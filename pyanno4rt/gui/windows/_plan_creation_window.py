"""Plan creation window."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from os.path import abspath, dirname, isfile
from PyQt5.QtCore import QDir, QEvent
from PyQt5.QtWidgets import QComboBox, QFileDialog, QMainWindow, QSpinBox

# %% Internal package import

from pyanno4rt.base import TreatmentPlan
from pyanno4rt.gui.compilations.plan_creation_window import (
    Ui_plan_creation_window)

# %% Class definition


class PlanCreationWindow(QMainWindow, Ui_plan_creation_window):
    """
    Plan creation window for the application.

    This class creates a plan creation window for the graphical user \
    interface, including input fields to declare a plan.
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

        # Install the custom event filter in the plan creation window
        self.new_plan_ref_cbox.installEventFilter(self)

        # Loop over the fieldnames with 'clicked' events
        for key, value in {
                'new_img_path_tbutton': self.add_imaging_path,
                'new_dose_path_tbutton': self.add_dose_matrix_path,
                }.items():

            # Connect the 'clicked' signal
            getattr(self, key).clicked.connect(value)

        # 
        self.new_plan_ledit.textChanged.connect(
            self.update_create_button)
        self.new_plan_ref_cbox.currentTextChanged.connect(
            self.update_create_button)
        self.new_img_path_ledit.textChanged.connect(
            self.update_create_button)
        self.new_dose_path_ledit.textChanged.connect(
            self.update_create_button)
        self.new_dose_res_ledit.textChanged.connect(
            self.update_create_button)

        # 
        self.new_plan_ref_cbox.currentTextChanged.connect(
            self.update_by_new_plan_reference)

        # 
        self.create_plan_pbutton.clicked.connect(self.create)

        # 
        self.close_plan_pbutton.clicked.connect(self.close)

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

    def position(self):
        """."""

        # Get the window geometry
        geometry = self.geometry()

        # Move the geometry center towards the parent
        geometry.moveCenter(self.parent.geometry().center())

        # Set the shifted geometry
        self.setGeometry(geometry)

    def update_create_button(self):
        """."""

        # 
        if (self.new_plan_ledit.text() in (
                self.parent.plan_select_cbox.itemText(i)
                for i in range(self.parent.plan_select_cbox.count()))
                or any(text == '' for text in (
                    self.new_plan_ledit.text(),
                    self.new_img_path_ledit.text(),
                    self.new_dose_path_ledit.text(),
                    self.new_dose_res_ledit.text()))):

            # 
            self.create_plan_pbutton.setEnabled(False)

        else:

            # 
            self.create_plan_pbutton.setEnabled(True)

        # 
        if (self.new_plan_ledit.text() != ''
                and self.new_plan_ref_cbox.currentText() != 'None'):

            # 
            self.create_plan_pbutton.setEnabled(True)

    def update_by_new_plan_reference(self):
        """."""

        # 
        if self.new_plan_ref_cbox.currentText() != 'None':

            # 
            self.new_modality_cbox.setEnabled(False)
            self.new_img_path_ledit.setEnabled(False)
            self.new_img_path_tbutton.setEnabled(False)
            self.new_dose_path_ledit.setEnabled(False)
            self.new_dose_path_tbutton.setEnabled(False)
            self.new_dose_res_ledit.setEnabled(False)

        else:

            # 
            self.new_modality_cbox.setEnabled(True)
            self.new_img_path_ledit.setEnabled(True)
            self.new_img_path_tbutton.setEnabled(True)
            self.new_dose_path_ledit.setEnabled(True)
            self.new_dose_path_tbutton.setEnabled(True)
            self.new_dose_res_ledit.setEnabled(True)

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
            self.new_img_path_ledit.setText(abspath(path))

            # Set the imaging path field cursor position to zero
            self.new_img_path_ledit.setCursorPosition(0)

    def add_dose_matrix_path(self):
        """Add the dose-influence matrix from a folder."""

        # Get the file path
        path, _ = QFileDialog.getOpenFileName(
            self, 'Select a dose-influence matrix file', QDir.rootPath(),
            'Dose-influence matrix (*.mat *.npy)')

        # Check if the file path exists
        if path:

            # Set the dose matrix path field
            self.new_dose_path_ledit.setText(abspath(path))

            # Set the dose matrix path field cursor position to zero
            self.new_dose_path_ledit.setCursorPosition(0)

    def create(self):
        """."""

        # 
        new_label = self.new_plan_ledit.text()
        reference = self.new_plan_ref_cbox.currentText()

        # 
        if reference != 'None':

            # Copy the reference input dictionaries
            configuration = self.parent.plans[reference].configuration.copy()
            optimization = self.parent.plans[reference].optimization.copy()
            evaluation = self.parent.plans[reference].evaluation.copy()

            # Change the treatment plan label
            configuration['label'] = new_label

            # Initialize the new treatment plan
            new_plan = TreatmentPlan(configuration, optimization, evaluation)

            # Activate the new treatment plan
            self.parent.activate(new_plan)

        else:

            # Reset the selector index to the default
            self.parent.plan_select_cbox.setCurrentIndex(-1)

            # 
            self.parent.plan_ledit.setText(new_label)

            # Set the treatment modality
            self.parent.modality_cbox.setCurrentText(
                self.new_modality_cbox.currentText())

            # Set the imaging path
            self.parent.img_path_ledit.setText(abspath(
                self.new_img_path_ledit.text()))

            # Set the dose matrix path
            self.parent.dose_path_ledit.setText(abspath(
                self.new_dose_path_ledit.text()))

            # Set the dose resolution
            self.parent.dose_res_ledit.setText(
                str(self.new_dose_res_ledit.text()))

            # 
            self.parent.initialize()

        # 
        self.parent.plan_ledit.setReadOnly(True)

        # 
        self.close()

    def close(self):
        """."""

        # 
        self.hide()
