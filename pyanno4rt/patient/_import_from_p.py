"""Python file import."""

# Author: Tim Ortkamp

# %% Internal package import

from pyanno4rt.patient._generate_ct_from_p import generate_ct_from_p
from pyanno4rt.patient._generate_segmentation_from_p import (
    generate_segmentation_from_p)
from pyanno4rt.patient._read_data_from_p import read_data_from_p

# %% Function definition


def import_from_p(path):
    """
    Import the patient data from a Python (.p) file.

    Parameters
    ----------
    path : str
        Path to the Python file.

    Returns
    -------
    dict
        Dictionary with information on the CT images.

    dict
        Dictionary with information on the segments.
    """

    # Read the CT and segmentation data
    computed_tomography_data, segmentation_data = read_data_from_p(path)

    # Generate the CT dictionary
    computed_tomography = generate_ct_from_p(computed_tomography_data)

    # Generate the segmentation dictionary
    segmentation = generate_segmentation_from_p(segmentation_data)

    return computed_tomography, segmentation
