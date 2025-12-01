"""MATLAB file handler."""

# Author: Tim Ortkamp

# %% External package import

from numpy import prod
from pandas import DataFrame
from scipy.io import loadmat, savemat

# %% Internal package import

from pyanno4rt.tools import filter_dict

# %% Class definition


class MatHandler():
    """
    MATLAB file handler class.

    This class provides methods to handle patient imaging data from MATLAB \
    files and generate the CT and segmentation dictionaries.

    Notes
    -----
    Only matRad- and pyanno4rt-style file formats are handled properly at the \
    moment. This will undergo thorough revision in a later release.
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
        """
        Save the patient imaging data.

        Parameters
        ----------
        computed_tomography : dict
            Dictionary with information on the CT images.

        segmentation : dict
            Dictionary with information on the segments.

        path : str
            Path for storing the patient imaging data.
        """

        # Copy the input dictionaries
        ct = computed_tomography.copy()
        cst = segmentation.copy()

        # Rearrange the cst as a dataframe
        cst = DataFrame.from_dict([{
            'index': values['index'],
            'segment': segment,
            'type': values['type'],
            'indices': values['raw_indices']+1,
            'parameters': values['parameters'],
            'components': []}
            for segment, values in cst.items()]).sort_values('index')

        # Save the data to a MATLAB file
        savemat(path, {'cst': cst, 'ct': ct})

    def generate_ct(
            self,
            data):
        """
        Generate the CT dictionary.

        Parameters
        ----------
        data : dict
            Raw data with information on the CT images.

        Returns
        -------
        dict
            Dictionary with information on the CT images.
        """

        # Initialize the CT dictionary with a subset of the data items
        computed_tomography = filter_dict(data, retain_keys=(
            'cubeHU', 'resolution', 'x', 'y', 'z', 'cube_dimensions',
            'cubeDim'))

        # Handle the different keys for the cube dimensions
        computed_tomography['cube_dimensions'] = computed_tomography.pop(
            'cube_dimensions', computed_tomography.get('cubeDim')).astype(int)

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
            Raw array with information on the segments.

        Returns
        -------
        dict
            Dictionary with information on the segments.
        """

        # Build a multi-layer tuple with the values per segment
        segment_values = (
            (row[1], ( # segment name
                row[0], # index
                row[2], # type
                row[3].astype(int)-1, # indices
                {f'{parameter[0].lower()}{parameter[1:]}': # parameters
                 value for parameter, value in row[4].__dict__.items()
                 if f'{parameter[0].lower()}{parameter[1:]}' in (
                         'priority', 'alphaX', 'betaX', 'visibleColor')},
                None, # objective
                None)) # constraint
            for row in data if len(row[3]) > 0)

        # Set the keys
        segment_keys = (
            'index', 'type', 'raw_indices', 'parameters', 'objective',
            'constraint')

        # Merge the keys and the values
        segmentation = {
            values[0]: dict(zip(segment_keys, values[1]))
            for values in segment_values}

        return dict(sorted(segmentation.items()))
