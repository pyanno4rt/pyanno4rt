"""MATLAB file-based CT dictionary generation."""

# Author: Tim Ortkamp

# %% External package import

from numpy import prod

# %% Internal package import

from pyanno4rt.tools import filter_dict

# %% Function definition


def generate_ct_from_mat(data):
    """
    Generate the CT dictionary from a MATLAB (.mat) file.

    Parameters
    ----------
    data : dict
        Dictionary with information on the CT slices.

    Returns
    -------
    dict
        Dictionary with information on the CT images.
    """

    # Initialize the CT dictionary with a subset of the data items
    computed_tomography = filter_dict(
        data, retain_keys=('cubeHU', 'resolution', 'x', 'y', 'z', 'cubeDim'))

    # Rename the cube dimensions key
    computed_tomography['cube_dimensions'] = (
        computed_tomography.pop('cubeDim').astype(int))

    # Add the number of voxels to the CT dictionary
    computed_tomography['number_of_voxels'] = prod(
        computed_tomography['cube_dimensions'])

    return computed_tomography
