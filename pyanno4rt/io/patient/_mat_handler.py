"""MATLAB file handler."""

# Author: Tim Ortkamp

# %% External package import

from numpy import prod
from scipy.io import loadmat

# %% Internal package import

from pyanno4rt.tools import filter_dict

# %% Class definition


class MatHandler():
    """
    MATLAB file handler class.

    This class provides methods to handle patient imaging data from MATLAB \
    files and generate the CT and segmentation dictionaries.
    """

    def __init__(self):

        pass

    def load(
            self,
            path):
        """
        Load the patient imaging data.

        Parameters
        ----------
        path : str
            Path to the patient imaging data.

        Returns
        -------
        dict
            Dictionary with information on the CT images.

        dict
            Dictionary with information on the segments.
        """

        # Load the file
        data = loadmat(path, simplify_cells=True)

        return (
            self.generate_ct(data['ct']),
            self.generate_segmentation(data['cst']))

    def save(
            self,
            computed_tomography,
            segmentation,
            path):
        """Save the patient imaging data."""

    def generate_ct(
            self,
            data):
        """
        Generate the CT dictionary.

        Parameters
        ----------
        data : dict
            Dictionary with information on the CT images.

        Returns
        -------
        dict
            Dictionary with information on the CT images.
        """

        # Initialize the CT dictionary with a subset of the data items
        computed_tomography = filter_dict(
            data,
            retain_keys=('cubeHU', 'resolution', 'x', 'y', 'z', 'cubeDim'))

        # Rename the cube dimensions key
        computed_tomography['cube_dimensions'] = (
            computed_tomography.pop('cubeDim').astype(int))

        # Add the number of voxels to the CT dictionary
        computed_tomography['number_of_voxels'] = prod(
            computed_tomography['cube_dimensions'])

        return computed_tomography

    def generate_segmentation(
            self,
            data):
        """
        Generate the segmentation dictionary.

        Parameters
        ----------
        data : ndarray
            Array with information on the segments.

        Returns
        -------
        dict
            Dictionary with information on the segments.
        """

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
                 if parameter in (
                         'Priority', 'alphaX', 'betaX', 'visibleColor')},
                None,
                None))
            for segment_values in data)

        # Set the dictionary keys
        segment_keys = (
            'index', 'type', 'raw_indices', 'prioritized_indices',
            'resized_indices', 'parameters', 'objective', 'constraint')

        # Merge the keys and the values into the segmentation dictionary
        segmentation = {
            values[0]: dict(zip(segment_keys, values[1]))
            for values in raw_segment_values}

        return dict(sorted(segmentation.items()))
