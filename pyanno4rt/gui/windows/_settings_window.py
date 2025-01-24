"""Settings window."""

# Author: Tim Ortkamp

# %% External package import

from json import loads
from PyQt5.QtWidgets import QMainWindow

# %% Internal package import

from pyanno4rt.gui.compilations.settings_window import Ui_settings_window
from pyanno4rt.gui.styles._custom_styles import pbutton_composer

# %% Class definition


class SettingsWindow(QMainWindow, Ui_settings_window):
    """
    Settings window for the GUI.

    This class sets up the settings window for the graphical user interface, \
    including some user-definable parameters.
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

        # 
        self.default = (
            'English', 'Dark', (1024, 768), (False, False, False, False))

        # 
        self.current = self.default

        # Temporarily disable combo boxes
        self.set_disabled(('language_cbox', 'light_mode_cbox'))

        # Set the stylesheets
        self.set_styles({
            'reset_pbutton': pbutton_composer,
            'save_pbutton': pbutton_composer,
            'close_pbutton': pbutton_composer})

        # Loop over the QComboBox elements in the settings window
        for box in ('language_cbox', 'light_mode_cbox', 'resolution_cbox'):

            # Install the custom event filters
            getattr(self, box).installEventFilter(parent)

        # Connect the fields with the event signals
        self.connect_signals()

    def set_disabled(
            self,
            field_names):
        """
        Disable multiple fields by their names.

        Parameters
        ----------
        field_names : tuple
            Tuple with the field names.
        """

        # Loop over the passed field names
        for name in field_names:

            # Get the attribute and disable the field
            getattr(self, name).setEnabled(False)

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

        # Loop over the field names with 'clicked' events
        for key, value in {
            'reset_pbutton': self.reset,
            'save_pbutton': self.save,
            'close_pbutton': self.close
                }.items():

            # Connect the 'clicked' event
            getattr(self, key).clicked.connect(value)

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
                    self.incl_model_data_check.isChecked(),
                    self.incl_opt_fluence_check.isChecked())

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
        self.incl_opt_fluence_check.setCheckState(2*settings[3][3])

    def reset(self):
        """."""

        self.set_fields(self.default)

    def save(self):
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
        self.parent.position()

        # 
        self.close()

    def position(self):
        """Set the window position."""

        # Get the window geometry
        geometry = self.geometry()

        # Move the geometry center according to the parent window
        geometry.moveCenter(self.parent.geometry().center())

        # Set the window geometry
        self.setGeometry(geometry)

    def close(self):
        """Close the settings window."""

        # Hide the window
        self.hide()
