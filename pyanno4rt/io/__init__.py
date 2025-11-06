"""
IO module.

==================================================================

This module aims to provide methods and classes for importing and processing \
external data, e.g. CT and segmentation data.
"""

# Author: Tim Ortkamp

from ._dicom_handler import DicomHandler
from ._mat_handler import MatHandler
from ._patient_loader import PatientLoader

__all__ = [
    'DicomHandler',
    'MatHandler',
    'PatientLoader']
