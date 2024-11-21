"""Splash screen window."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from importlib.metadata import version
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QGraphicsDropShadowEffect, QMainWindow, QDesktopWidget)

# %% Internal package import

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

        # Add the version label
        self.version_label.setText(
            ' '.join(('Amadeus', version('pyanno4rt'))))

        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)

        self.shadow = QGraphicsDropShadowEffect(self)
        self.shadow.setBlurRadius(30)
        self.shadow.setXOffset(0)
        self.shadow.setYOffset(0)
        self.shadow.setColor(QColor(0, 0, 0, 150))
        self.splash_widget.setGraphicsEffect(self.shadow)

    def position(self):
        """."""

        qtRectangle = self.frameGeometry()
        centerPoint = QDesktopWidget().availableGeometry().center()
        qtRectangle.moveCenter(centerPoint)
        self.move(qtRectangle.topLeft())
