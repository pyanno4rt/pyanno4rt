"""Text window."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from PyQt5.QtWidgets import QMainWindow

# %% Internal package import

from pyanno4rt.gui.compilations.text_window import Ui_text_window

# %% Class definition


class TextWindow(QMainWindow, Ui_text_window):
    """
    Text window for the application.

    This class creates a text window for the graphical user interface, \
    including a scrollable text box for display.
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
        self.close_text_pbutton.clicked.connect(self.close)

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
