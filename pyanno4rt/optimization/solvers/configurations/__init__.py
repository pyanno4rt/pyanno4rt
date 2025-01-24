"""
Solution algorithms module.

==================================================================

This module aims to provide functions to configure the solution algorithms \
for the optimization packages.
"""

# Author: Tim Ortkamp

from ._configure_ipyopt import configure_ipyopt
from ._configure_proxmin import configure_proxmin
# from ._configure_pyanno4rt import configure_pyanno4rt
from ._configure_pymoo import configure_pymoo
from ._configure_pypop7 import configure_pypop7
from ._configure_scipy import configure_scipy

__all__ = ['configure_ipyopt',
           'configure_proxmin',
           # 'configure_pyanno4rt',
           'configure_pymoo',
           'configure_pypop7',
           'configure_scipy']
