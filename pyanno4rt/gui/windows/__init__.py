"""
GUI windows module.

==================================================================

The module aims to provide methods and classes to ...
"""

# Author: Tim Ortkamp

# Import the subwindows
from pyanno4rt.gui.windows._compare_window import CompareWindow
from pyanno4rt.gui.windows._data_columns_window import DataColumnsWindow
from pyanno4rt.gui.windows._info_window import InfoWindow
from pyanno4rt.gui.windows._log_window import LogWindow
from pyanno4rt.gui.windows._plan_creation_window import PlanCreationWindow
from pyanno4rt.gui.windows._settings_window import SettingsWindow
from pyanno4rt.gui.windows._splash_screen_window import SplashScreenWindow
from pyanno4rt.gui.windows._text_window import TextWindow
from pyanno4rt.gui.windows._tree_window import TreeWindow

# Import the main window
from pyanno4rt.gui.windows._main_window import MainWindow

# Import the component windows
from pyanno4rt.gui.windows import components

__all__ = [
    'components',
    'CompareWindow',
    'DataColumnsWindow',
    'InfoWindow',
    'LogWindow',
    'MainWindow',
    'PlanCreationWindow',
    'SettingsWindow',
    'SplashScreenWindow',
    'TextWindow',
    'TreeWindow']
