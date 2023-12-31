"""Segmentation dictionary generation from MATLAB (.mat) file."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from numpy import ravel_multi_index, unravel_index, where, zeros
from scipy.ndimage import zoom

# %% Function definition


def generate_segmentation_from_mat(data, ct_dict, resolution):
    """
    Generate the segmentation dictionary.

    Parameters
    ----------
    data : ndarray
        Array with information on the segmented structures.

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

    # Build a multi-layer tuple with the values for the dictionary
    segment_values = ((
        segment[1],
        (
            segment[0],
            segment[2],
            segment[3].astype(int)-1,
            segment[3].astype(int)-1,
            segment[3].astype(int)-1,
            {''.join(
                (parameter[0].lower(),
                 parameter[1:])): segment[4].__dict__[parameter]
             for parameter in segment[4].__dict__ if parameter in (
                     'Priority', 'alphaX', 'betaX', 'visibleColor')},
            None,
            None
         )) for segment in data)

    # Set the dictionary keys
    segment_keys = ('index', 'type', 'raw_indices',
                    'prioritized_indices', 'resized_indices',
                    'parameters', 'objective', 'constraint')

    # Merge the keys and the values into the segmentation dictionary
    segmentation = {segment_values[0]:
                    dict(zip(segment_keys, segment_values[1]))
                    for segment_values in segment_values}

    # Check if interpolation should be performed
    if all(key in (*ct_dict,) for key in ('old_dimensions', 'zooms')):

        # Return the interpolated segmentation dictionary
        return interpolate_segmentation_dictionary()

    return segmentation
