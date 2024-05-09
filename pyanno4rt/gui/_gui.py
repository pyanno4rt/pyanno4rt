"""Graphical user interface."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from pyqtgraph import mkQApp

# %% Internal package import

from pyanno4rt.gui.windows import MainWindow

# %% Class definition


class GraphicalUserInterface():
    """
    Graphical user interface class.

    This class provides ...

    Parameters
    ----------
    ...

    Attributes
    ----------
    ...
    """

    def __init__(self):

        # Initialize the application
        self.application = mkQApp("Graphical User Interface")

        # Set the application style
        self.application.setStyle('Fusion')

    def launch(
            self,
            plan=None):
        """Launch the graphical user interface."""

        # Initialize the main window for the GUI
        self.main_window = MainWindow(plan, self.application)

        # Run the application
        self.main_window.application.exec_()

    def fetch(self):
        """Get the treatment plan dictionary of the GUI."""

        return self.main_window.plans

    def closeEvent(
            self,
            event):
        """
        Close the application.

        Parameters
        ----------
        event : object of class `QCloseEvent`
            Instance of the class `QCloseEvent` to be triggered at window \
            closing.
        """

        # Close the application
        self.application.quit()
