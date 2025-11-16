"""
Graphical user interface module.

==================================================================

The module aims to provide methods and classes to ...
"""

# Author: Tim Ortkamp

# Import the graphical user interface
from ._graphical_user_interface import GraphicalUserInterface

# Import the submodules
from . import custom_widgets, windows

# Import the resources file
from .assets import resources_rc

__all__ = [
    'GraphicalUserInterface',
    'custom_widgets',
    'windows',
    'resources_rc']
