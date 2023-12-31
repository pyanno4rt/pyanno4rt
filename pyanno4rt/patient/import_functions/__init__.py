"""
Import functions module.

==================================================================

The module aims to provide methods for importing patient data from different \
input formats, i.e., CT data (from patient imaging) and segmentation data \
(from image segmentation), and automatically converting them to appropriate \
dictionary formats.
"""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

from ._generate_ct_from_dcm import generate_ct_from_dcm
from ._generate_ct_from_mat import generate_ct_from_mat
from ._generate_ct_from_p import generate_ct_from_p

from ._generate_segmentation_from_dcm import generate_segmentation_from_dcm
from ._generate_segmentation_from_mat import generate_segmentation_from_mat
from ._generate_segmentation_from_p import generate_segmentation_from_p

from ._import_from_dcm import import_from_dcm
from ._import_from_mat import import_from_mat
from ._import_from_p import import_from_p

from ._read_data_from_dcm import read_data_from_dcm
from ._read_data_from_mat import read_data_from_mat
from ._read_data_from_p import read_data_from_p

__all__ = ['generate_ct_from_dcm',
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
           'read_data_from_p']
