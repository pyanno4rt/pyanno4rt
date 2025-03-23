"""
Patient module.

==================================================================

This module aims to provide methods and classes for importing and processing \
patient data, i.e., CT and segmentation data.
"""

# Author: Tim Ortkamp

# Import the CT functions
from ._generate_ct_from_dcm import generate_ct_from_dcm
from ._generate_ct_from_mat import generate_ct_from_mat
from ._generate_ct_from_p import generate_ct_from_p

# Import the segmentation functions
from ._generate_segmentation_from_dcm import generate_segmentation_from_dcm
from ._generate_segmentation_from_mat import generate_segmentation_from_mat
from ._generate_segmentation_from_p import generate_segmentation_from_p

# Import the data reading functions
from ._read_data_from_dcm import read_data_from_dcm
from ._read_data_from_mat import read_data_from_mat
from ._read_data_from_p import read_data_from_p

# Import the data import functions
from ._import_from_dcm import import_from_dcm
from ._import_from_mat import import_from_mat
from ._import_from_p import import_from_p

# Import the patient loader
from ._patient_loader import PatientLoader

__all__ = [
    'generate_ct_from_dcm',
    'generate_ct_from_mat',
    'generate_ct_from_p',
    'generate_segmentation_from_dcm',
    'generate_segmentation_from_mat',
    'generate_segmentation_from_p',
    'import_from_dcm',
    'import_from_mat',
    'import_from_p',
    'read_data_from_dcm',
    'read_data_from_mat',
    'read_data_from_p',
    'PatientLoader']
