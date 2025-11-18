"""
Patient IO module.

==================================================================

This module aims to provide methods and classes for importing and exporting \
the patient data (CT/segmentation).
"""

# Author: Tim Ortkamp

from ._dicom_handler import DicomHandler
from ._mat_handler import MatHandler

__all__ = [
    'DicomHandler',
    'MatHandler']
