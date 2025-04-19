"""
Visualization module.

==================================================================

The module aims to provide methods and classes to visualize different aspects \
of the generated treatment plans, with respect to optimization problem \
analysis, data-driven model review, and treatment plan evaluation.
"""

# Author: Tim Ortkamp

from pyanno4rt.visualization import static_plots
from pyanno4rt.visualization._visualization_window import VisualizationWindow

__all__ = [
    'static_plots',
    'VisualizationWindow']
