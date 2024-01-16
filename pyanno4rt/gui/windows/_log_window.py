"""Logging window."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from PyQt5.QtWidgets import QMainWindow

# %% Internal package import

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

        # Get the application from the argument
        self.parent = parent

        # Run the constructor from the superclass
        super().__init__()

        # Build the UI main window
        self.setupUi(self)

        # 
        self.close_log_pbutton.clicked.connect(self.close)

    def position(self):
        """."""

        # Get the window geometry
        geometry = self.geometry()

        # Move the geometry center towards the parent
        geometry.moveCenter(self.parent.geometry().center())

        # Set the shifted geometry
        self.setGeometry(geometry)

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

    def close(self):
        """."""

        # 
        self.hide()
