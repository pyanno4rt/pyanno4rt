"""
Covariance matrix adaptation evolution strategy (CMAES) module.

==================================================================

This module aims to provide methods and classes for CMAES solvers.
"""

# Author: Tim Ortkamp

from ._cmaes import CMAES
from ._low_rank_integrator import LowRankIntegrator
from ._lrcmaes import LRCMAES

__all__ = [
    'CMAES',
    'LowRankIntegrator',
    'LRCMAES']
