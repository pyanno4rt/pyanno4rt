"""DICOM data reading."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from os import listdir
from pydicom import dcmread

# %% Function definition


def read_data_from_dcm(path):
    """
    Read the DICOM data from the path.

    Parameters
    ----------
    path : str
        Path to the DICOM folder.

    Returns
    -------
    computed_tomography_data : tuple
        Tuple of :class:`pydicom.dataset.FileDataset` objects with \
        information on the CT slices.

    segmentation_data : object of class :class:`pydicom.dataset.FileDataset`
        The object representation of the segmentation data.
    """

    # Load the DICOM files
    files = tuple(dcmread(f'{path}{file}') for file in listdir(path))

    # Get the (axially ordered) CT data files
    computed_tomography_data = tuple(sorted(
        [file for file in files if hasattr(file, 'PixelData')],
        key=lambda file: file.ImagePositionPatient[2]))

    # Get the segmentation data file
    segmentation_data = next(
        file for file in files if hasattr(file, 'ROIContourSequence'))

    return computed_tomography_data, segmentation_data
