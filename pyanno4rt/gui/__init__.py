"""
Graphical user interface module.

==================================================================

The module aims to provide methods and classes to ...
"""

# Author: Tim Ortkamp

# Import the graphical user interface
from ._gui import GraphicalUserInterface

# Import the submodules
from . import custom_widgets
from . import windows

# Import the resources file
from .assets import resources_rc

__all__ = [
    'GraphicalUserInterface',
    'custom_widgets',
    'windows',
    'resources_rc']
