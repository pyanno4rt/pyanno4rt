"""Plan creation window."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from PyQt5.QtWidgets import QMainWindow

# %% Internal package import

from pyanno4rt.base import TreatmentPlan
from pyanno4rt.gui.compilations.plan_creation_window import (
    Ui_plan_create_window)

# %% Class definition


class PlanCreationWindow(QMainWindow, Ui_plan_create_window):
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

        # 
        self.new_plan_ledit.textChanged.connect(
            self.update_by_new_plan_label)

        # 
        self.create_plan_pbutton.clicked.connect(self.create)

        # 
        self.close_plan_pbutton.clicked.connect(self.close)

    def position(self):
        """."""

        # Get the window geometry
        geometry = self.geometry()

        # Move the geometry center towards the parent
        geometry.moveCenter(self.parent.geometry().center())

        # Set the shifted geometry
        self.setGeometry(geometry)

    def update_by_new_plan_label(self):
        """."""

        # 
        if self.new_plan_ledit.text() in (
                self.parent.plan_select_cbox.itemText(i)
                for i in range(self.parent.plan_select_cbox.count())):

            # 
            self.create_plan_pbutton.setEnabled(False)

        else:

            # 
            self.create_plan_pbutton.setEnabled(True)

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

        # 
        self.parent.plan_ledit.setReadOnly(True)

        # 
        self.close()

    def close(self):
        """."""

        # 
        self.hide()
