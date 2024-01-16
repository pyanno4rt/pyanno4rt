"""Settings window."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from json import loads
from PyQt5.QtWidgets import QMainWindow

# %% Internal package import

from pyanno4rt.gui.compilations.settings_window import Ui_settings_window

# %% Class definition


class SettingsWindow(QMainWindow, Ui_settings_window):
    """
    Settings window for the GUI.

    This class creates the settings window for the graphical user interface, \
    including some user-definable parameters.
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
        self.default = ('English', 'Dark', (1920, 1080), (False, False, False))
        self.current = self.default

        # Temporarily disable combo boxes
        self.language_cbox.setEnabled(False)
        self.light_mode_cbox.setEnabled(False)

        # 
        self.reset_settings_pbutton.clicked.connect(self.reset)
        self.save_settings_pbutton.clicked.connect(self.save_apply_close)

    def position(self):
        """."""

        # Get the window geometry
        geometry = self.geometry()

        # Move the geometry center towards the parent
        geometry.moveCenter(self.parent.geometry().center())

        # Set the shifted geometry
        self.setGeometry(geometry)

    def get_fields(self):
        """."""

        # 
        language = self.language_cbox.currentText()

        # 
        light_mode = self.light_mode_cbox.currentText()

        # 
        resolution = tuple(
            map(loads, self.resolution_cbox.currentText().split('x')))

        # 
        includes = (self.incl_img_data_check.isChecked(),
                    self.incl_dij_check.isChecked(),
                    self.incl_model_data_check.isChecked())

        return (language, light_mode, resolution, includes)

    def set_fields(self, settings):
        """."""

        # 
        self.language_cbox.setCurrentText(settings[0])

        # 
        self.light_mode_cbox.setCurrentText(settings[1])

        # 
        self.resolution_cbox.setCurrentIndex(0)
        self.resolution_cbox.setCurrentText('x'.join(map(str, settings[2])))

        # 
        self.incl_img_data_check.setCheckState(2*settings[3][0])
        self.incl_dij_check.setCheckState(2*settings[3][1])
        self.incl_model_data_check.setCheckState(2*settings[3][2])

    def reset(self):
        """."""

        self.set_fields(self.default)

    def save_apply_close(self):
        """."""

        # 
        self.current = self.get_fields()

        #
        if self.parent.isMaximized() and self.current[2][0] <= 1440:

            # Show the main window in normal mode
            self.parent.showNormal()

            # Resize the main window
            self.parent.resize(*self.current[2])

        elif (self.current[2][0] >=
              self.parent.application.primaryScreen().size().width()):

            # 
            self.parent.showMaximized()

        else:

            # Resize the main window
            self.parent.resize(*self.current[2])

        # 
        self.hide()
