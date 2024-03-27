"""
Patient module.

==================================================================

This module aims to provide methods and classes for importing and processing \
patient data.
"""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

from . import import_functions

from ._patient_loader import PatientLoader

__all__ = ['import_functions',
           'PatientLoader']
