"""
Visualization module.

==================================================================

The module aims to provide methods and classes to visualize different aspects \
of the generated treatment plans, with respect to optimization problem \
analysis, data-driven model review, and treatment plan evaluation.
"""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

from pyanno4rt.visualization._visualizer import Visualizer
from pyanno4rt.visualization import visuals

__all__ = ['Visualizer',
           'visuals']
