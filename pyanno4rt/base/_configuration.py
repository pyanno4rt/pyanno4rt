"""Configuration handler."""

# Author: Tim Ortkamp

# %% External package import

from os.path import abspath
from functools import partial

# %% Internal package import

from pyanno4rt.checking import (
    check_length, check_regular_extension, check_regular_extension_directory,
    check_subtype, check_type, check_value, check_value_in_set)
from pyanno4rt.tools import filter_dict

# %% Class definition


class Configuration():
    """
    Configuration handler class.

    This class provides methods to handle the configuration parameters of a \
    treatment plan.

    Parameters
    ----------
    label : str
        Label for the treatment plan.

        .. note:: To prevent overwriting processes, choose a unique label for \
            each treatment plan!
        .. note:: To prevent memory issues, keep the label unchanged if \
            possible, once set!

    modality : {'photon', 'proton'}
        Treatment modality.

        .. note::
            - modality='photon': \
            :class:`~pyanno4rt.optimization.projections._dose_projection.DoseProjection`\
            with neutral RBE of 1.0
            - modality='proton': \
            :class:`~pyanno4rt.optimization.projections._constant_rbe_projection.ConstantRBEProjection`\
            with constant RBE of 1.1

    imaging_path : str
        Path to the CT and segmentation data (.dcm, .mat or .p).

        .. note::
            Requirements:

            - Matlab/Python files should include 'ct' and 'cst' as variables
            - DICOM folders should include a series of CT files and one \
                structure file

    dose_matrix_path : str
        Path to the dose-influence matrix file (.mat, .npy or .npz).

    dose_resolution : list
        Size of the dose grid in `[mm]` per dimension.

    min_log_level : {'debug', 'info', 'warning', 'error, 'critical'}, \
                     default='info'
        Minimum logging level.

    number_of_fractions : int, default=30
        Number of fractions according to the treatment scheme.

    Attributes
    ----------
    label : str
        See 'Parameters'.

    modality : {'photon', 'proton'}
        See 'Parameters'.

    imaging_path : str
        See 'Parameters'.

    dose_matrix_path : str
        See 'Parameters'.

    dose_resolution : list
        See 'Parameters'.

    min_log_level : {'debug', 'info', 'warning', 'error', 'critical'}
        See 'Parameters'.

    number_of_fractions : int
        See 'Parameters'.
    """

    def __init__(
            self,
            label,
            modality,
            imaging_path,
            dose_matrix_path,
            dose_resolution,
            min_log_level='info',
            number_of_fractions=30):

        # Get the input arguments
        inputs = filter_dict(vars(), remove_keys=('self',))

        # Check the input arguments
        self.check(inputs)

        # Loop over the input arguments
        for key, value in inputs.items():

            # Set the attribute
            setattr(self, key, value)

        # Convert the paths into absolute values
        self.imaging_path = abspath(self.imaging_path)
        self.dose_matrix_path = abspath(self.dose_matrix_path)

    def to_dict(self):
        """Serialize the object into a dictionary."""

        return vars(self)

    @classmethod
    def from_dict(
            cls,
            dictionary):
        """
        Deserialize the object from a dictionary.

        Parameters
        ----------
        dictionary : dict
            Dictionary with the configuration parameters.

        Returns
        -------
        object of class :class:`~pyanno4rt.base._configuration.Configuration`
            The object used to handle the plan configuration parameters.
        """

        return cls(**dictionary)

    def check(
            self,
            inputs):
        """
        Check the input arguments.

        Parameters
        ----------
        inputs : dict
            Dictionary with the input arguments.
        """

        # Get the check map
        check_map = {
            'label': (
                partial(check_type, types=str),
                partial(check_length, reference=1, sign='>=')),
            'modality': (
                partial(check_type, types=str),
                partial(check_value_in_set, options=('photon', 'proton'))),
            'imaging_path': (
                partial(check_type, types=str),
                partial(check_regular_extension, extensions=('.mat', '.p')),
                partial(check_regular_extension_directory, extensions=(
                    '.dcm',), no_directory=('.mat', '.p'))),
            'dose_matrix_path': (
                partial(check_type, types=str),
                partial(check_regular_extension, extensions=(
                    '.mat', '.npy', 'npz'))),
            'dose_resolution': (
                partial(check_type, types=list),
                partial(check_subtype, types=(int, float)),
                partial(check_length, reference=3, sign='=='),
                partial(check_value, reference=1, sign='>=', is_vector=True)),
            'min_log_level': (
                partial(check_type, types=str),
                partial(check_value_in_set, options=(
                    'debug', 'info', 'warning', 'error', 'critical'))),
            'number_of_fractions': (
                partial(check_type, types=int),
                partial(check_value, reference=1, sign='>='))}

        # Loop over the inputs
        for key, value in inputs.items():

            # Loop over the check functions
            for function in check_map[key]:

                # Run the check function
                function(key, value)
