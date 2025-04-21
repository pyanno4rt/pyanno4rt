"""Logging window."""

# Author: Tim Ortkamp

# %% External package import

from PyQt5.QtWidgets import QMainWindow

# %% Internal package import

from pyanno4rt.gui._custom_styles import pbutton_composer
from pyanno4rt.gui.compilations.log_window import Ui_log_window

# %% Class definition


class LogWindow(QMainWindow, Ui_log_window):
    """
    Logging window for the application.

    This class creates the log window for the graphical user interface, \
    including the output of the logger.
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
            'close_log_pbutton': pbutton_composer})

        # Connect the fields with the event signals
        self.close_log_pbutton.clicked.connect(self.close)

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
        self.close_log_pbutton.clicked.connect(self.close)

    def update_log_output(self):
        """."""

        # 
        self.log_tedit.clear()

        # 
        instance = self.parent.plans[self.parent.plan_ledit.text()]

        # 
        stream_value = instance.logger.logger.handlers[1].stream.getvalue()

        # 
        stream_value = stream_value.replace('\n', '\n\n')

        # 
        self.log_tedit.setText(stream_value)

    def position(self):
        """Set the window position."""

        # Get the window geometry
        geometry = self.geometry()

        # Move the geometry center according to the parent window
        geometry.moveCenter(self.parent.geometry().center())

        # Set the window geometry
        self.setGeometry(geometry)

    def close(self):
        """Close the log window."""

        # Hide the window
        self.hide()
