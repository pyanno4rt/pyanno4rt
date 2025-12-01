"""
Logging module.

==================================================================

This module aims to provide methods and classes to track events during \
pyanno4rt execution cycles.
"""

# Author: Tim Ortkamp

from ._logging import Logging
from ._logging_utils import get_logger, set_logger_name

__all__ = [
    'Logging',
    'get_logger',
    'set_logger_name']
