"""
Check maps module.

==================================================================

This module aims to provide scripts with mappings between the members of \
different input parameter groups and their validity check functions.
"""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

from ._component_map import component_map
from ._configuration_map import configuration_map
from ._evaluation_map import evaluation_map
from ._model_display_map import model_display_map
from ._model_map import model_map
from ._optimization_map import optimization_map
from ._top_level_map import top_level_map
from ._tune_space_map import tune_space_map

__all__ = ['component_map',
           'configuration_map',
           'evaluation_map',
           'model_display_map',
           'model_map',
           'optimization_map',
           'top_level_map',
           'tune_space_map']
