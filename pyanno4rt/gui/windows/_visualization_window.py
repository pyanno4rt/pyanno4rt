"""Visualization window."""

# Author: Tim Ortkamp

# %% External package import

from PyQt5.QtWidgets import QMainWindow

# %% Internal package import

from pyanno4rt.gui.compilations.visualization_window import Ui_vis_window
from pyanno4rt.gui.styles._custom_styles import pbutton_composer

# %% Class definition


class VisualizationWindow(QMainWindow, Ui_vis_window):
    """
    Visualization window for the GUI.

    This class creates the visualization window for the graphical user \
    interface, including different types of interactive and static plots.
    """

    def __init__(
            self,
            parent=None):

        # Run the constructor from the superclass
        super().__init__()

        # Build the UI main window
        self.setupUi(self)

        # Get the application from the argument
        self.parent = parent

        # Set the stylesheets
        self.set_styles({
            'close_visualizer_pbutton': pbutton_composer})

        # Connect the fields with the event signals
        self.connect_signals()

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

    def connect_signals(self):
        """Connect the fields with the event signals."""

        # Connect the 'clicked' event with the close button
        self.close_visualizer_pbutton.clicked.connect(self.close)

    def position(self):
        """Set the window position."""

        # Get the window geometry
        geometry = self.geometry()

        # Move the geometry center according to the parent window
        geometry.moveCenter(self.parent.geometry().center())

        # Set the window geometry
        self.setGeometry(geometry)

    def close(self):
        """Close the visualization window."""

        # Hide the window
        self.hide()
