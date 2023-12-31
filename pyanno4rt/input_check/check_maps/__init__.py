"""
Check maps module.

==================================================================

The module aims to provide files with the mappings between different types \
of dictionaries and the checking functions to be executed.
"""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

from ._top_level_map import top_level_map
from ._configuration_map import configuration_map
from ._optimization_map import optimization_map
from ._evaluation_map import evaluation_map
from ._objective_map import objective_map
from ._model_map import model_map

__all__ = ['top_level_map',
           'configuration_map',
           'optimization_map',
           'evaluation_map',
           'objective_map',
           'model_map']
