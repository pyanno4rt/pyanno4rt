"""Segmentation dictionary generation from DICOM (.dcm) file."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from colorsys import hsv_to_rgb
from numpy import (append, array, column_stack, ravel_multi_index, sort,
                   where, zeros)
from scipy.interpolate import interp1d
from skimage.draw import polygon2mask

# %% Function definition


def generate_segmentation_from_dcm(data, ct_data, ct_dict):
    """
    Generate the segmentation dictionary.

    Parameters
    ----------
    data : object of class `FileDataset`
        Instance of the class `FileDataset`, which contains information on \
        the segmented structures.

    ct_data : tuple
        Tuple of 'FileDataset' instances with information on the CT slices.

    ct_dict : dict
        Dictionary with information on the CT images.

    Returns
    -------
    segmentation : dict
        Dictionary with information on the segmented structures.
    """

    def convert_hsv_to_rgb(hue, saturation, value):
        """Convert colors from the HSV to the RBG space."""

        return hsv_to_rgb(hue, saturation, value)

    def generate_colors(length):
        """Generate a tuple of specific length with different RGB colors."""

        return tuple(array(convert_hsv_to_rgb(value/(length + 1), 1.0, 1.0))
                     for value in range(length))

    def compute_segment_indices():
        """Compute the (sorted) binary segment indices."""

        # Initialize the segment cube
        segment_cube = zeros(ct_dict['cube_dimensions'])

        # Loop over the contour sequences
        for sequence in roi_contour.ContourSequence._list:

            # Check if the geometric type is different from 'POINT'
            if sequence.ContourGeometricType != 'POINT':

                # Get the grid points of the sequence
                points_x = sequence.ContourData._list[0::3]
                points_y = sequence.ContourData._list[1::3]
                points_z = sequence.ContourData._list[2::3]

                # Check if the endpoints are different from the starting points
                if (points_x[-1] != points_x[0]
                        or points_y[-1] != points_y[0]
                        or points_z[-1] != points_z[0]):

                    # Close the contour polygon by adding the starting points
                    points_x = append(points_x, points_x[0])
                    points_y = append(points_y, points_y[0])
                    points_z = append(points_z, points_z[0])

                # Round the z-points to account for numerical issues
                points_z = [1e-10 * round(1e10 * value) for value in points_z]

                # Check if contour points outside the slice exist
                if len(set(points_z)) > 1:

                    # Raise an error to indicate out-of-slice points
                    raise ValueError(
                        f"The contour sequence for the segment {segment} "
                        "includes out-of-slice points!")

                # Check if CT data exists for the current contour slice
                if min(ct_dict['z']) <= points_z[0] <= max(ct_dict['z']):

                    # Interpolate the axis points in x
                    interpolated_x = interp1d(
                        ct_dict['x'], range(ct_dict['cube_dimensions'][1]),
                        'linear', fill_value='extrapolate')(points_x)

                    # Interpolate the axis points in y
                    interpolated_y = interp1d(
                        ct_dict['y'], range(ct_dict['cube_dimensions'][0]),
                        'linear', fill_value='extrapolate')(points_y)

                    # Convert the polygon vertices into a binary mask
                    mask = polygon2mask(
                        ct_dict['cube_dimensions'][:2],
                        column_stack((interpolated_y, interpolated_x))
                        + (0, 0.5))

                    # Get the CT slice indices for the current sequence
                    ct_slice_indices = [
                        index for index, value in enumerate(ct_dict['z'])
                        if (points_z[0] - int(ct_data[0].SliceThickness)/2
                            <= value
                            < points_z[0] + int(ct_data[0].SliceThickness)/2)]

                    # Loop over the slice indices
                    for index in ct_slice_indices:

                        # Enter the binary mask into the segment cube
                        segment_cube[:, :, index] = mask

        # Get the segment indices
        segment_indices = ravel_multi_index(
            where(segment_cube == 1),
            ct_dict['cube_dimensions'], order='F')

        return sort(segment_indices)

    # Get the default color tuple
    default_colors = generate_colors(len(data.ROIContourSequence._list))

    # Initialize the segmentation dictionary
    segmentation = {}

    # Loop over the ROI contours
    for roi_contour in data.ROIContourSequence._list:

        # Find the corresponding ROI structure from the index number
        roi_structure = next(
            sequence for sequence in data.StructureSetROISequence._list
            if roi_contour.ReferencedROINumber == sequence.ROINumber)

        # Get the structure name
        segment = roi_structure.ROIName

        # Add the first-layer backbone to the dictionary
        segmentation[segment] = {
            key: None
            for key in ('index', 'type', 'raw_indices', 'prioritized_indices',
                        'resized_indices', 'parameters', 'objective',
                        'constraint')
            }

        # Add the second-layer backbone to the dictionary
        segmentation[segment]['parameters'] = {
            key: None
            for key in ('priority', 'alphaX', 'betaX', 'visibleColor')}

        # Add the segment index to the dictionary
        segmentation[segment]['index'] = int(roi_contour.ReferencedROINumber)-1

        # Check if the segment is a target
        if any(string in segment.lower() for string in (
                'tv', 'target', 'gtv', 'ctv', 'ptv', 'boost', 'tumor')):

            # Add the 'TARGET' type to the dictionary
            segmentation[segment]['type'] = 'TARGET'

            # Add the default target priority to the dictionary
            segmentation[segment]['parameters']['priority'] = 1

        else:

            # Add the 'OAR' type to the dictionary
            segmentation[segment]['type'] = 'OAR'

            # Add the default organ-at-risk priority to the dictionary
            segmentation[segment]['parameters']['priority'] = 2

        # Add the default biological parameters to the dictionary
        segmentation[segment]['parameters']['alphaX'] = 0.1
        segmentation[segment]['parameters']['betaX'] = 0.05

        # Check if the ROI contour includes a display color
        if hasattr(roi_contour, 'ROIDisplayColor'):

            # Add the visible color to the dictionary
            segmentation[segment]['parameters']['visibleColor'] = array([
                int(number) / 255
                for number in roi_contour.ROIDisplayColor._list])

        else:

            # Add the default visible color to the dictionary
            segmentation[segment]['parameters']['visibleColor'] = (
                default_colors[segmentation[segment]['index']])

        # Check if the ROI contour includes contour sequence data
        if (hasattr(roi_contour, 'ContourSequence')
                and roi_contour.ContourSequence):

            # Add the segment indices to the dictionary
            segmentation[segment]['raw_indices'] = compute_segment_indices()

    return dict(sorted(segmentation.items()))
