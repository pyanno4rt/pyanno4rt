"""DICOM folder import."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% Internal package import

from pyanno4rt.patient.import_functions._generate_ct_from_dcm import (
    generate_ct_from_dcm)
from pyanno4rt.patient.import_functions._generate_segmentation_from_dcm import (
    generate_segmentation_from_dcm)
from pyanno4rt.patient.import_functions._read_data_from_dcm import (
    read_data_from_dcm)

# %% Function definition


def import_from_dcm(path, resolution):
    """
    Import the patient data from a folder with DICOM (.dcm) files.

    Parameters
    ----------
    path : str
        Path to the DICOM folder.

    resolution : None or list
        Imaging resolution for post-processing interpolation of the CT and \
        segmentation data.

    Returns
    -------
    dict
        Dictionary with information on the CT images.

    dict
        Dictionary with information on the segmented structures.
    """

    # Read the CT and segmentation data
    computed_tomography_data, segmentation_data = read_data_from_dcm(path)

    # Generate the CT dictionary
    computed_tomography = generate_ct_from_dcm(
        computed_tomography_data, resolution)

    # Generate the segmentation dictionary
    segmentation = generate_segmentation_from_dcm(
        segmentation_data, computed_tomography_data, computed_tomography)

    return computed_tomography, segmentation
