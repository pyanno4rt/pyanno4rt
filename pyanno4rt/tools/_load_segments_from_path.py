"""Segment names and types loading."""

# Author: Tim Ortkamp

# %% External package import

from os import listdir
from os.path import splitext

from pydicom import dcmread
from scipy.io import loadmat

# %% Function definition


def load_segments_from_path(path):
    """
    Return the segment names and types from a path.

    Parameters
    ----------
    path : str
        Path to the CT and segmentation data.

    Returns
    -------
    dict
        Dictionary with the segment names and types.
    """

    # Set the target flags
    flags = ('tv', 'target', 'gtv', 'ctv', 'ptv', 'boost', 'tumor')

    # Check if the path leads to a DICOM folder
    if splitext(path)[1] == '':

        # Load the DICOM files
        files = tuple(dcmread(f'{path}/{file}') for file in listdir(path))

        # Get the segmentation data file
        segmentation_data = next(
            file for file in files if hasattr(file, 'ROIContourSequence'))

        # Initialize the segment dictionary
        segments = {}

        # Loop over the ROI contours
        for roi_contour in segmentation_data.ROIContourSequence:

            # Find the corresponding segment from the index number
            roi_structure = next(
                sequence
                for sequence in segmentation_data.StructureSetROISequence
                if roi_contour.ReferencedROINumber == sequence.ROINumber)

            # Check if the segment is a target volume
            if any(string in roi_structure.ROIName.lower()
                   for string in flags):

                # Set the segment type to 'TARGET'
                segment_type = 'TARGET'

            else:

                # Set the segment type to 'OAR'
                segment_type = 'OAR'

            # Add segment name and type to the dictionary
            segments |= {roi_structure.ROIName: segment_type}

    # Else, check if the path leads to a MATLAB file
    elif splitext(path)[1] == '.mat':

        # Load the MATLAB segmentation data
        segmentation_data = loadmat(path, simplify_cells=True)['cst']

        # Generate the segment dictionary
        segments = {
            segment_values[1]: segment_values[2]
            for segment_values in segmentation_data}

    else:

        # Set the segment dictionary empty
        segments = {}

    return dict(sorted(segments.items()))
