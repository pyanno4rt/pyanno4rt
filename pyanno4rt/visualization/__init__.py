"""
Visualization module.

==================================================================

The module aims to provide methods and classes to visualize different aspects \
of the generated treatment plans, with respect to optimization problem \
analysis, data-driven model review, and treatment plan evaluation.
"""

# Author: Tim Ortkamp

# Import the visualizer
from pyanno4rt.visualization._visualizer import Visualizer

# Import the submodules
from . import custom_widgets, static

# Import the resources file
from .assets import resources_rc

__all__ = [
    'Visualizer'
    'custom_widgets',
    'static',
    'resources_rc']
