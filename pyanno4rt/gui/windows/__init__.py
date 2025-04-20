"""
GUI windows module.

==================================================================

The module aims to provide methods and classes to ...
"""

# Author: Tim Ortkamp

# Import the subwindows
from ._compare_window import CompareWindow
from ._data_columns_window import DataColumnsWindow
from ._info_window import InfoWindow
from ._log_window import LogWindow
from ._plan_creation_window import PlanCreationWindow
from ._settings_window import SettingsWindow
from ._splash_screen_window import SplashScreenWindow
from ._text_window import TextWindow
from ._tree_window import TreeWindow

# Import the main window
from ._main_window import MainWindow

# Import the component windows
from . import components

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
