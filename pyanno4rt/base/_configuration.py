"""Plan configuration information."""

# Author: Tim Ortkamp

# %% External package import

from functools import partial

# %% Internal package import

from pyanno4rt.input_check.check_functions import (
    check_length, check_regular_extension, check_regular_extension_directory,
    check_subtype, check_type, check_value, check_value_in_set)
from pyanno4rt.tools import filter_dict

# %% Class definition


class Configuration():
    """
    Plan configuration information class.

    This class provides methods to set, validate and serialize the \
    configuration parameters of the treatment plan.

    Parameters
    ----------
    label : str
        Unique identifier for the treatment plan.

        .. note:: Uniqueness of the label is important because it \
            prevents overwriting processes between different treatment \
            plan instances by isolating their datahubs, logging channels \
            and general storage paths.
        .. note:: Changing the label of a treatment plan instance will \
            automatically create a new singleton datahub object. To \
            prevent memory issues, keep the label unchanged if possible, \
            once set!

    modality : {'photon', 'proton'}
        Treatment modality, needs to be consistent with the dose \
        calculation inputs.

        .. note:: If the modality is 'photon', \
            :class:`~pyanno4rt.optimization.projections._dose_projection.DoseProjection`\
            with neutral RBE of 1.0 is automatically applied, whereas for \
            the modality 'proton', \
            :class:`~pyanno4rt.optimization.projections._constant_rbe_projection.ConstantRBEProjection`\
            with constant RBE of 1.1 is used.

    imaging_path : str
        Path to the CT and segmentation data.

        .. note:: It is assumed that CT and segmentation data are \
            included in a single file (.mat or .p) or a series of files \
            (.dcm), whose content follows the pyanno4rt data structure.

    dose_matrix_path : str
        Path to the dose-influence matrix file (.mat, .npy or .npz).

    dose_resolution : list
        Size of the dose grid in [`mm`] per dimension, needs to be \
        consistent with the dose calculation inputs.

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

        # Check the input arguments
        self.check(filter_dict(vars(), remove_keys=('self',)))

        # Loop over the input arguments
        for key, value in filter_dict(vars(), remove_keys=('self',)).items():

            # Set the attribute
            setattr(self, key, value)

    def to_dict(self):
        """Return the configuration parameter dictionary."""

        return vars(self)

    def check(
            self,
            input_dictionary):
        """
        Check the items of an input dictionary.

        Parameters
        ----------
        input_dictionary : dict
            Dictionary with the mappings between parameter names and values.
        """

        # Get the check map
        check_map = {
            'label': (partial(check_type, types=str),),
            'min_log_level': (
                partial(check_type, types=str),
                partial(check_value_in_set, options=(
                    'debug', 'info', 'warning', 'error', 'critical'))),
            'modality': (
                partial(check_type, types=str),
                partial(check_value_in_set, options=('photon', 'proton'))),
            'number_of_fractions': (
                partial(check_type, types=int),
                partial(check_value, reference=0, sign='>')),
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
                partial(check_value, reference=0, sign='>', is_vector=True))}

        # Loop over the dictionary items
        for key, value in input_dictionary.items():

            # Loop over the check functions
            for function in check_map[key]:

                # Run the check function
                function(key, value)
