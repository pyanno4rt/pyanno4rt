"""
Graphical user interface module.

==================================================================

The module aims to provide methods and classes to ...
"""

# Author: Tim Ortkamp

# Import the graphical user interface
from pyanno4rt.gui._gui import GUI

# Import the submodules
from pyanno4rt.gui import custom_widgets, windows

# Import the resources file
from pyanno4rt.gui.assets import resources_rc

__all__ = [
    'GUI',
    'custom_widgets',
    'windows',
    'resources_rc']
