"""DICOM (.dcm) data reading."""

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
    path : string
        Path to the DICOM folder.

    Returns
    -------
    computed_tomography_data : tuple
        Tuple of 'FileDataset' instances with information on the CT slices.

    segmentation_data : object of class `FileDataset`
        Instance of the class `FileDataset`, which contains information on \
        the segmented structures.
    """

    # Load the DICOM files
    files = [dcmread(''.join((path, filename)))
             for filename in listdir(path)]

    # Get the (sorted) CT data files
    computed_tomography_data = tuple(sorted(
        [file for file in files if hasattr(file, 'PixelData')],
        key=lambda x: x.ImagePositionPatient._list[2]))

    # Get the segmentation data file
    segmentation_data = next(
        file for file in files if hasattr(file, 'ROIContourSequence'))

    return computed_tomography_data, segmentation_data
