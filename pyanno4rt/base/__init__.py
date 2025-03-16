"""
Base module.

==================================================================

This module aims to provide base classes to generate treatment plans.
"""

# Author: Tim Ortkamp

from ._configuration import Configuration
from ._evaluation import Evaluation
from ._optimization import Optimization
from ._treatment_plan import TreatmentPlan

__all__ = [
    'Configuration',
    'Evaluation',
    'Optimization',
    'TreatmentPlan']
