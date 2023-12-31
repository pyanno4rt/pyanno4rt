"""Segmentation dictionary generation from Python (.p) file."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from numpy import ravel_multi_index, unravel_index, where, zeros
from scipy.ndimage import zoom

# %% Function definition


def generate_segmentation_from_p(data, ct_dict, resolution):
    """
    Generate the segmentation dictionary.

    Parameters
    ----------
    data : dict
        Dictionary with information on the segmented structures.

    ct_dict : dict
        Dictionary with information on the CT images.

    resolution : None or list
        Imaging resolution for post-processing interpolation of the CT and \
        segmentation data.

    Returns
    -------
    segmentation : dict
        Dictionary with information on the segmented structures.
    """

    def interpolate_segmentation_dictionary():
        """Interpolate the segmentation dictionary to the target resolution."""

        # Loop over the segments
        for segment in (*segmentation,):

            # Initialize the segment mask
            mask = zeros(ct_dict['old_dimensions'])

            # Insert ones at the indices of the segment
            mask[unravel_index(
                segmentation[segment]['raw_indices'],
                ct_dict['old_dimensions'], order='F')] = 1

            # Get the resized indices of the segment
            resized_indices = where(zoom(mask, ct_dict['zooms'], order=0))

            # Enter the resized indices into the datahub
            segmentation[segment]['raw_indices'] = ravel_multi_index(
                resized_indices, ct_dict['cube_dimensions'], order='F')

        return segmentation

    # Initialize the segmentation dictionary directly by the data
    segmentation = data

    # Check if interpolation should be performed
    if all(key in (*ct_dict,) for key in ('old_dimensions', 'zooms')):

        # Return the interpolated segmentation dictionary
        return interpolate_segmentation_dictionary()

    return segmentation
