"""DICOM folder import."""

# Author: Tim Ortkamp

# %% Internal package import

from pyanno4rt.patient import generate_ct_from_dcm
from pyanno4rt.patient import generate_segmentation_from_dcm
from pyanno4rt.patient import read_data_from_dcm

# %% Function definition


def import_from_dcm(path):
    """
    Import the patient data from a folder with DICOM (.dcm) files.

    Parameters
    ----------
    path : str
        Path to the DICOM folder.

    Returns
    -------
    dict
        Dictionary with information on the CT images.

    dict
        Dictionary with information on the segments.
    """

    # Read the CT and segmentation data
    computed_tomography_data, segmentation_data = read_data_from_dcm(path)

    # Generate the CT dictionary
    computed_tomography = generate_ct_from_dcm(computed_tomography_data)

    # Generate the segmentation dictionary
    segmentation = generate_segmentation_from_dcm(
        segmentation_data, computed_tomography_data, computed_tomography)

    return computed_tomography, segmentation
