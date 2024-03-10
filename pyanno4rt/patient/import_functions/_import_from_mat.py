"""MATLAB file import."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% Internal package import

from pyanno4rt.patient.import_functions._generate_ct_from_mat import (
    generate_ct_from_mat)
from pyanno4rt.patient.import_functions._generate_segmentation_from_mat import (
    generate_segmentation_from_mat)
from pyanno4rt.patient.import_functions._read_data_from_mat import (
    read_data_from_mat)

# %% Function definition


def import_from_mat(path, resolution):
    """
    Import the patient data from a MATLAB (.mat) file.

    Parameters
    ----------
    path : str
        Path to the MATLAB file.

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
    computed_tomography_data, segmentation_data = read_data_from_mat(path)

    # Generate the CT dictionary
    computed_tomography = generate_ct_from_mat(
        computed_tomography_data, resolution)

    # Generate the segmentation dictionary
    segmentation = generate_segmentation_from_mat(
        segmentation_data, computed_tomography)

    return {key: value for key, value in computed_tomography.items()
            if key not in ('old_dimensions', 'zooms')}, segmentation
