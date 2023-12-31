"""
Patient data module.

==================================================================

The module aims to provide methods and classes for importing patient data, \
automatically converting them to appropriate dictionary formats, and storing \
them in the datahub.
"""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

from ._patient_loader import PatientLoader

from . import import_functions

__all__ = ['PatientLoader',
           'import_functions']
