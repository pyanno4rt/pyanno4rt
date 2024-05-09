"""Python file-based segmentation dictionary generation."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from numpy import ravel_multi_index, unravel_index, where, zeros
from scipy.ndimage import zoom

# %% Function definition


def generate_segmentation_from_p(data, computed_tomography):
    """
    Generate the segmentation dictionary from a Python binary (.p) file.

    Parameters
    ----------
    data : dict
        Dictionary with information on the segmented structures.

    computed_tomography : dict
        Dictionary with information on the CT images.

    Returns
    -------
    dict
        Dictionary with information on the segmented structures.
    """

    def interpolate_segmentation_dictionary(segmentation, computed_tomography):
        """Interpolate the segmentation dictionary to a resolution."""

        # Loop over the segments
        for segment in segmentation:

            # Initialize the segment mask
            mask = zeros(computed_tomography['old_dimensions'])

            # Insert ones at the segment indices
            mask[unravel_index(
                segmentation[segment]['raw_indices'],
                computed_tomography['old_dimensions'], order='F')] = 1

            # Get the resized segment indices
            resized_indices = where(
                zoom(mask, computed_tomography['zooms'], order=0))

            # Enter the new segment indices into the datahub
            segmentation[segment]['raw_indices'] = ravel_multi_index(
                resized_indices, computed_tomography['cube_dimensions'],
                order='F')

        return segmentation

    # Check if the interpolation parameters are available
    if all(key in computed_tomography for key in ('old_dimensions', 'zooms')):

        # Return the interpolated segmentation dictionary
        return interpolate_segmentation_dictionary(
            data, computed_tomography)

    return data
