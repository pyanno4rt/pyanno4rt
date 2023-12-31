"""
Datahub module.

==================================================================

The module aims to provide a singleton implementation for centralizing the \
data structures generated across the pyanno4rt program to efficiently manage \
and distribute information units.
"""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

from ._datahub import Datahub

__all__ = ['Datahub']
