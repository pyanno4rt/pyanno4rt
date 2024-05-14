"""Plan comparison window."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from PyQt5.QtWidgets import QMainWindow

# %% Internal package import

from pyanno4rt.gui.compilations.compare_window import Ui_compare_window
from pyanno4rt.gui.custom_widgets._slice_compare_widget import (
    SliceCompareWidget)

# %% Class definition


class CompareWindow(QMainWindow, Ui_compare_window):
    """
    Plan comparison window for the GUI.

    This class creates the plan comparison window for the graphical user \
    interface, including some general information on the package.
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
        self.baseline_slice_widget = SliceCompareWidget(self)
        self.reference_slice_widget = SliceCompareWidget(self)

        # 
        self.baseline_layout.insertWidget(0, self.baseline_slice_widget)
        self.reference_layout.insertWidget(0, self.reference_slice_widget)

        # 
        self.slice_selection_sbar.valueChanged.connect(
            self.baseline_slice_widget.change_image_slice)
        self.slice_selection_sbar.valueChanged.connect(
            self.reference_slice_widget.change_image_slice)

        # 
        self.opacity_sbox.valueChanged.connect(
            self.baseline_slice_widget.change_dose_opacity)
        self.opacity_sbox.valueChanged.connect(
            self.reference_slice_widget.change_dose_opacity)

        # 
        self.close_info_pbutton.clicked.connect(self.close)

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
