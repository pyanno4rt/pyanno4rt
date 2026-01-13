"""Configuration handler."""

# Author: Tim Ortkamp

# %% External package import

from os.path import abspath
from functools import partial

# %% Internal package import

from pyanno4rt.tools import filter_dict
from pyanno4rt.validation import (
    validate_directory, validate_file, validate_item, validate_item_in_set,
    validate_length, validate_subtype, validate_type)

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
        Path to the CT and segmentation data (.dcm or .mat).

        .. note::
            Requirements:

            - Matlab files should include 'ct' and 'cst' as variables, \
                with a structure similar to matRad (https://e0404.github.io/matRad/)
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

        # Check if an imaging path has been provided
        if imaging_path is not None:

            # Convert into an absolute path
            imaging_path = abspath(imaging_path)

        # Check if a dose-influence matrix path has been provided
        if dose_matrix_path is not None:

            # Convert into an absolute path
            dose_matrix_path = abspath(dose_matrix_path)

        # Validate the input arguments
        self.validate(filter_dict(vars(), remove_keys=('self',)))

        # Get the input attributes
        self.label = label
        self.modality = modality
        self.imaging_path = imaging_path
        self.dose_matrix_path = dose_matrix_path
        self.dose_resolution = dose_resolution
        self.min_log_level = min_log_level
        self.number_of_fractions = number_of_fractions

    def to_dict(self):
        """
        Serialize the configuration handler into a dictionary.

        Returns
        -------
        dict
            Dictionary with the configuration handler's arguments.
        """

        return vars(self)

    @classmethod
    def from_dict(
            cls,
            dictionary):
        """
        Deserialize the configuration handler from a dictionary.

        Parameters
        ----------
        dictionary : dict
            Dictionary with the configuration handler's arguments.

        Returns
        -------
        object of class :class:`~pyanno4rt.base._configuration.Configuration`
            The object used to handle the plan configuration parameters.
        """

        return cls(**dictionary)

    def validate(
            self,
            inputs):
        """
        Validate the input arguments.

        Parameters
        ----------
        inputs : dict
            Dictionary with the input arguments.
        """

        # Get the validation map
        validation_map = {
            'label': (
                partial(validate_type, options=str),
                partial(validate_length, reference=1, sign='>=')
                ),
            'modality': (
                partial(validate_type, options=str),
                partial(validate_item_in_set, options=('photon', 'proton'))
                ),
            'imaging_path': (
                partial(validate_type, options=str),
                partial(validate_file, options=('.mat',)),
                partial(validate_directory, options=('.dcm',), alt=('.mat',))
                ),
            'dose_matrix_path': (
                partial(validate_type, options=str),
                partial(validate_file, options=('.mat', '.npy', 'npz'))
                ),
            'dose_resolution': (
                partial(validate_type, options=list),
                partial(validate_subtype, options=(int, float)),
                partial(validate_length, reference=3, sign='=='),
                partial(validate_item, reference=1, sign='>=')
                ),
            'min_log_level': (
                partial(validate_type, options=str),
                partial(validate_item_in_set, options=(
                    'debug', 'info', 'warning', 'error', 'critical'))
                ),
            'number_of_fractions': (
                partial(validate_type, options=int),
                partial(validate_item, reference=1, sign='>=')
                )
            }

        # Loop over the inputs
        for key, value in inputs.items():

            # Loop over the validation functions
            for function in validation_map[key]:

                # Run the validation function
                function(key, value)
