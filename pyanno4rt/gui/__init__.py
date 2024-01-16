"""
Graphical user interface module.

==================================================================

The module aims to provide methods and classes to ...
"""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

from ._gui import GraphicalUserInterface
from . import custom_widgets
from . import windows
from .assets import resources_rc

__all__ = ['GraphicalUserInterface',
           'custom_widgets',
           'windows',
           'resources_rc']
