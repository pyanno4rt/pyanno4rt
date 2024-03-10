"""MATLAB file-based CT dictionary generation."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from numpy import array, prod
from scipy.ndimage import zoom

# %% Internal package import

from pyanno4rt.tools import arange_with_endpoint

# %% Function definition


def generate_ct_from_mat(data, resolution):
    """
    Generate the CT dictionary from a MATLAB (.mat) file.

    Parameters
    ----------
    data : dict
        Dictionary with information on the CT slices.

    resolution : None or list
        Imaging resolution for post-processing interpolation of the CT and \
        segmentation data.

    Returns
    -------
    computed_tomography : dict
        Dictionary with information on the CT images.
    """

    def interpolate_ct_dictionary(computed_tomography, resolution):
        """Interpolate the CT dictionary values to a resolution."""

        # Get the current cube dimensions
        old_dimensions = computed_tomography['cube_dimensions']

        # Update the grid resolution
        computed_tomography['resolution'] = dict(
            zip(('x', 'y', 'z'), resolution))

        # Loop over the grid axes
        for index, axis in enumerate(('x', 'y', 'z')):

            # Update the grid points on the current axis
            computed_tomography[axis] = arange_with_endpoint(
                computed_tomography[axis][0],
                computed_tomography[axis][-1],
                resolution[index])

        # Update the cube dimensions
        computed_tomography['cube_dimensions'] = array([
            len(computed_tomography[axis]) for axis in ('x', 'y', 'z')])

        # Get the zoom factors for all cube dimensions
        zooms = tuple(pair[0]/pair[1] for pair in zip(
            computed_tomography['cube_dimensions'], old_dimensions))

        # Interpolate the CT cube to the target resolution
        computed_tomography['cube'] = zoom(
            computed_tomography['cube'], zooms, order=1)

        # Update the number of voxels
        computed_tomography['number_of_voxels'] = prod(
            computed_tomography['cube_dimensions'])

        # Add the interpolation information for the segmentation
        computed_tomography['old_dimensions'] = old_dimensions
        computed_tomography['zooms'] = zooms

        return computed_tomography

    # Initialize the CT dictionary with a subset of the data items
    computed_tomography = {key: value for key, value in data.items()
                           if key in ('cube', 'resolution', 'x', 'y', 'z',
                                      'cubeDim')}

    # Rename the cube dimensions key
    computed_tomography['cube_dimensions'] = (
        computed_tomography.pop('cubeDim').astype(int))

    # Add the number of voxels to the CT dictionary
    computed_tomography['number_of_voxels'] = prod(
        computed_tomography['cube_dimensions'])

    # Check if a target resolution has been passed
    if resolution:

        # Return the interpolated CT dictionary
        return interpolate_ct_dictionary(computed_tomography, resolution)

    return computed_tomography
