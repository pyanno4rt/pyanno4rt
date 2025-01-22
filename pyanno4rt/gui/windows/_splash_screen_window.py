"""Splash screen window."""

# Author: Tim Ortkamp

# %% External package import

from time import sleep

from importlib.metadata import version
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QIcon
from PyQt5.QtWidgets import (
    QApplication, QGraphicsDropShadowEffect, QMainWindow)

# %% Internal package import

from pyanno4rt.gui.assets import resources_rc
from pyanno4rt.gui.compilations.splash_screen_window import Ui_splash_window

# %% Class definition


class SplashScreenWindow(QMainWindow, Ui_splash_window):
    """
    Splash screen window for the GUI.

    This class sets up the splash screen window for the graphical user \
    interface.
    """

    def __init__(self):

        # Run the constructor from the superclass
        super().__init__()

        # Build the UI main window
        self.setupUi(self)

        # Set the window icon
        self.setWindowIcon(QIcon(
            ':/special_icons/icons_special/logo_white_icon.png'))

        # Add the version label
        self.version_label.setText(f'"Amadeus" v{version("pyanno4rt")}')

        # Set the window flags
        self.setWindowFlags(Qt.FramelessWindowHint)

        # Set the background attribute
        self.setAttribute(Qt.WA_TranslucentBackground)

        # Initialize the shadow effect
        self.shadow = QGraphicsDropShadowEffect(self)

        # Set the blur radius
        self.shadow.setBlurRadius(0)

        # Set the offset
        self.shadow.setXOffset(0)
        self.shadow.setYOffset(0)

        # Set the color
        self.shadow.setColor(QColor(0, 0, 0, 100))

        # Add the shadow effect to the widget
        self.splash_widget.setGraphicsEffect(self.shadow)

    def progress(self):
        """Trigger the progress bar."""

        # Loop over the value range
        for i in range(1, 101):

            # Sleep
            sleep(0.03)

            # Set the value
            self.progressBar.setValue(i)

        # Sleep
        sleep(1)

    def position(self):
        """Set the window position."""

        # Get the frame geometry
        geometry = self.frameGeometry()

        # Get the screen number from the cursor position
        screen = QApplication.desktop().screenNumber(
            QApplication.desktop().cursor().pos())

        # Move the geometry center according to the application window
        geometry.moveCenter(
            QApplication.desktop().screenGeometry(screen).center())

        # Move the window to the top left of the geometry
        self.move(geometry.topLeft())
