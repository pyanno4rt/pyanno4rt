"""DICOM folder-based segmentation dictionary generation."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from colorsys import hsv_to_rgb
from numpy import (
    append, array, column_stack, ravel_multi_index, sort, where, zeros)
from scipy.interpolate import interp1d
from skimage.draw import polygon2mask

# %% Function definition


def generate_segmentation_from_dcm(data, ct_slices, computed_tomography):
    """
    Generate the segmentation dictionary from a folder with DICOM (.dcm) files.

    Parameters
    ----------
    data : object of class :class:`pydicom.dataset.FileDataset`
        The :class:`pydicom.dataset.FileDataset` object with information on \
        the segmented structures.

    slices : tuple
        Tuple of :class:`pydicom.dataset.FileDataset` objects with \
        information on the CT slices.

    computed_tomography : dict
        Dictionary with information on the CT images.

    Returns
    -------
    segmentation : dict
        Dictionary with information on the segmented structures.

    Raises
    ------
    ValueError
        If the contour sequence for a segment includes out-of-slice points.
    """

    def generate_colors(length):
        """Generate a tuple of specific length with different RGB colors."""

        return tuple(array(hsv_to_rgb(value/(length+1), 1.0, 1.0))
                     for value in range(length))

    def compute_segment_indices(ct_slices, computed_tomography, roi_contour):
        """Compute the (sorted) binary segment indices."""

        # Initialize the segment cube
        segment_cube = zeros(computed_tomography['cube_dimensions'])

        # Loop over the contour sequences
        for sequence in roi_contour.ContourSequence:

            # Check if the geometric type is different from 'POINT'
            if sequence.ContourGeometricType != 'POINT':

                # Get the grid points of the sequence
                points_x, points_y, points_z = (
                    sequence.ContourData[i::3] for i in range(3))

                # Loop over the grid point dimensions
                for points in (points_x, points_y, points_z):

                    # Check if the endpoint differs from the starting point
                    if points[-1] != points[0]:

                        # Close the contour polygon by adding the first point
                        points = append(points, points[0])

                # Round the z-points to account for numerical issues
                points_z = [1e-10*round(1e10*value) for value in points_z]

                # Check if contour points outside the slice exist
                if len(set(points_z)) > 1:

                    # Raise an error to indicate out-of-slice points
                    raise ValueError(
                        f"The contour sequence for the segment {segment} "
                        "includes out-of-slice points!")

                # Check if the current contour slice lies within the data
                if (min(computed_tomography['z'])
                        <= points_z[0]
                        <= max(computed_tomography['z'])):

                    # Interpolate the points on the x- and y-axis
                    interpolated_x, interpolated_y = (interp1d(
                        computed_tomography[axis[0]],
                        range(computed_tomography['cube_dimensions'][axis[1]]),
                        'linear', fill_value='extrapolate')(axis[2])
                        for axis in (('x', 1, points_x), ('y', 0, points_y)))

                    # Convert the polygon vertices into a binary mask
                    mask = polygon2mask(
                        computed_tomography['cube_dimensions'][:2],
                        column_stack((interpolated_y, interpolated_x))
                        + (0, 0.5))

                    # Get the computed tomography slice indices
                    ct_slice_indices = [
                        index
                        for index, value in enumerate(computed_tomography['z'])
                        if (points_z[0]-int(ct_slices[0].SliceThickness)/2
                            <= value
                            < points_z[0]+int(ct_slices[0].SliceThickness)/2)]

                    # Loop over the slice indices
                    for index in ct_slice_indices:

                        # Enter the binary mask into the segment cube
                        segment_cube[:, :, index] = mask

        return sort(ravel_multi_index(
            where(segment_cube == 1), computed_tomography['cube_dimensions'],
            order='F'))

    # Get the default color tuple
    default_colors = generate_colors(len(data.ROIContourSequence))

    # Initialize the segmentation dictionary
    segmentation = {}

    # Loop over the ROI contours
    for roi_contour in data.ROIContourSequence:

        # Find the corresponding ROI structure from the index number
        roi_structure = next(
            sequence for sequence in data.StructureSetROISequence
            if roi_contour.ReferencedROINumber == sequence.ROINumber)

        # Get the structure name
        segment = roi_structure.ROIName

        # Add the first-layer backbone to the dictionary
        segmentation[segment] = {
            key: None for key in (
                'index', 'type', 'raw_indices', 'prioritized_indices',
                'resized_indices', 'parameters', 'objective', 'constraint')}

        # Add the second-layer backbone to the dictionary
        segmentation[segment]['parameters'] = {
            key: None for key in (
                'priority', 'alphaX', 'betaX', 'visibleColor')}

        # Add the segment index to the dictionary
        segmentation[segment]['index'] = int(roi_contour.ReferencedROINumber)-1

        # Check if the segment is a target volume
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
            segmentation[segment]['parameters']['visibleColor'] = array(
                [int(num)/255 for num in roi_contour.ROIDisplayColor])

        else:

            # Add the default visible color to the dictionary
            segmentation[segment]['parameters']['visibleColor'] = (
                default_colors[segmentation[segment]['index']])

        # Check if the ROI contour includes contour sequence data
        if (hasattr(roi_contour, 'ContourSequence')
                and roi_contour.ContourSequence):

            # Add the segment indices to the dictionary
            segmentation[segment]['raw_indices'] = compute_segment_indices(
                ct_slices, computed_tomography, roi_contour)

    return dict(sorted(segmentation.items()))
