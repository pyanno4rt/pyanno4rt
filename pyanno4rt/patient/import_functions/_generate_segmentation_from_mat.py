"""MATLAB file-based segmentation dictionary generation."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from numpy import ravel_multi_index, unravel_index, where, zeros
from scipy.ndimage import zoom

# %% Function definition


def generate_segmentation_from_mat(data, computed_tomography):
    """
    Generate the segmentation dictionary from a MATLAB (.mat) file.

    Parameters
    ----------
    data : ndarray
        Array with information on the segmented structures.

    computed_tomography : dict
        Dictionary with information on the CT images.

    Returns
    -------
    segmentation : dict
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

    # Build a multi-layer tuple with the values for the dictionary
    raw_segment_values = (
        (segment_values[1], (
            segment_values[0],
            segment_values[2],
            segment_values[3].astype(int)-1,
            segment_values[3].astype(int)-1,
            segment_values[3].astype(int)-1,
            {f'{parameter[0].lower()}{parameter[1:]}':
             segment_values[4].__dict__[parameter]
             for parameter in segment_values[4].__dict__
             if parameter in ('Priority', 'alphaX', 'betaX', 'visibleColor')},
            None,
            None))
        for segment_values in data)

    # Set the dictionary keys
    segment_keys = ('index', 'type', 'raw_indices', 'prioritized_indices',
                    'resized_indices', 'parameters', 'objective', 'constraint')

    # Merge the keys and the values into the segmentation dictionary
    segmentation = {values[0]: dict(zip(segment_keys, values[1]))
                    for values in raw_segment_values}

    # Check if the interpolation parameters are available
    if all(key in computed_tomography for key in ('old_dimensions', 'zooms')):

        # Return the interpolated segmentation dictionary
        return interpolate_segmentation_dictionary(
            segmentation, computed_tomography)

    return segmentation
