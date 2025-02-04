"""Segment names and types loading."""

# Author: Tim Ortkamp

# %% External package import

from os.path import splitext

# %% Internal package import

from pyanno4rt.patient.import_functions import (
    read_data_from_dcm, read_data_from_mat, read_data_from_p)

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

        # Read the DICOM segmentation data
        _, segmentation_data = read_data_from_dcm(path)

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

    # Check if the path leads to a MATLAB file
    elif splitext(path)[1] == '.mat':

        # Read the MATLAB segmentation data
        _, segmentation_data = read_data_from_mat(path)

        # Generate the segment dictionary
        segments = {
            segment_values[1]: ('TARGET' if any(
                string in segment_values[1].lower() for string in flags)
                else 'OAR') for segment_values in segmentation_data}

    # Check if the path leads to a Python file
    elif splitext(path)[1] == '.p':

        # Get the segmentation data
        _, segmentation_data = read_data_from_p(path)

        # Get the segments
        segments = {
            segment: ('TARGET' if any(
                string in segment.lower() for string in flags)
                else 'OAR') for segment in segmentation_data}

    else:

        # Set the segment dictionary empty
        segments = {}

    return dict(sorted(segments.items()))
