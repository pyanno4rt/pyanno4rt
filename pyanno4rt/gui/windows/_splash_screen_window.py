"""Splash screen window."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from importlib.metadata import version
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QIcon
from PyQt5.QtWidgets import (
    QGraphicsDropShadowEffect, QMainWindow, QDesktopWidget)
from time import sleep

# %% Internal package import

from pyanno4rt.gui.assets import resources_rc
from pyanno4rt.gui.compilations.splash_screen_window import Ui_splash_window

# %% Class definition


class SplashScreenWindow(QMainWindow, Ui_splash_window):
    """
    Splash screen window for the application.

    This class creates a splash screen window for the graphical user interface.
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

        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)

        self.shadow = QGraphicsDropShadowEffect(self)
        self.shadow.setBlurRadius(70)
        self.shadow.setXOffset(0)
        self.shadow.setYOffset(0)
        self.shadow.setColor(QColor(0, 0, 0, 100))
        self.splash_widget.setGraphicsEffect(self.shadow)

    def progress(self):
        """."""

        # 
        for i in range(1, 101):

            # 
            sleep(0.03)

            # 
            self.progressBar.setValue(i)

        # 
        sleep(0.3)

    def position(self):
        """."""

        qtRectangle = self.frameGeometry()
        centerPoint = QDesktopWidget().availableGeometry().center()
        qtRectangle.moveCenter(centerPoint)
        self.move(qtRectangle.topLeft())
