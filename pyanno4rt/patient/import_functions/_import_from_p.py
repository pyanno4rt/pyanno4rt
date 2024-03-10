"""Python file import."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% Internal package import

from pyanno4rt.patient.import_functions._generate_ct_from_p import (
    generate_ct_from_p)
from pyanno4rt.patient.import_functions._generate_segmentation_from_p import (
    generate_segmentation_from_p)
from pyanno4rt.patient.import_functions._read_data_from_p import (
    read_data_from_p)

# %% Function definition


def import_from_p(path, resolution):
    """
    Import the patient data from a Python (.p) file.

    Parameters
    ----------
    path : str
        Path to the Python file.

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
    computed_tomography_data, segmentation_data = read_data_from_p(path)

    # Generate the CT dictionary
    computed_tomography = generate_ct_from_p(
        computed_tomography_data, resolution)

    # Generate the segmentation dictionary
    segmentation = generate_segmentation_from_p(
        segmentation_data, computed_tomography)

    return {key: value for key, value in computed_tomography.items()
            if key not in ('old_dimensions', 'zooms')}, segmentation
