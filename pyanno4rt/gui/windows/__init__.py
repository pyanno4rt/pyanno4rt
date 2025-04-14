"""
GUI windows module.

==================================================================

The module aims to provide methods and classes to ...
"""

# Author: Tim Ortkamp

from ._compare_window import CompareWindow
from ._data_columns_window import DataColumnsWindow
from ._info_window import InfoWindow
from ._log_window import LogWindow
from ._plan_creation_window import PlanCreationWindow
from ._settings_window import SettingsWindow
from ._splash_screen_window import SplashScreenWindow
from ._text_window import TextWindow
from ._tree_window import TreeWindow
from ._visualization_window import VisualizationWindow

from ._main_window import MainWindow

__all__ = [
    'CompareWindow',
    'DataColumnsWindow',
    'InfoWindow',
    'LogWindow',
    'MainWindow',
    'PlanCreationWindow',
    'SettingsWindow',
    'SplashScreenWindow',
    'TextWindow',
    'TreeWindow',
    'VisualizationWindow']
