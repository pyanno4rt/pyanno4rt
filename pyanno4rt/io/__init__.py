"""
IO module.

==================================================================

This module aims to provide methods and classes for handling input and output \
streams, e.g. for CT/segmentation and dose data.
"""

# Author: Tim Ortkamp

from . import dose_matrix, patient

__all__ = [
    'dose_matrix',
    'patient']
