"""Evaluation handler."""

# Author: Tim Ortkamp

# %% External package import

from functools import partial

# %% Internal package import

from pyanno4rt.tools import filter_dict
from pyanno4rt.validation import (
    validate_item, validate_item_in_set, validate_subtype, validate_type)

# %% Class definition


class Evaluation():
    """
    Evaluation handler class.

    This class provides methods to handle the evaluation parameters of a \
    treatment plan.

    Parameters
    ----------
    dvh_type : {'cumulative', 'differential'}, default=cumulative'
        Type of DVH to be evaluated.

    number_of_points : int, default=1000
        Number of (evenly-spaced) DVH evaluation points.

    reference_volumes : tuple or list, default=(2, 5, 50, 95, 98)
        Reference volumes for the inverse DVH values.

    reference_doses : tuple or list, default=()
        Reference doses for the DVH values.

        .. note:: If the default value is used, reference dose levels will be \
            determined automatically.

    Attributes
    ----------
    dvh_type : {'cumulative', 'differential'}
        See 'Parameters'.

    number_of_points : int
        See 'Parameters'.

    reference_volumes : tuple or list
        See 'Parameters'.

    reference_doses : tuple or list
        See 'Parameters'.
    """

    def __init__(
            self,
            dvh_type='cumulative',
            number_of_points=1000,
            reference_volumes=(2, 5, 50, 95, 98),
            reference_doses=()):

        # Validate the input arguments
        self.validate(filter_dict(vars(), remove_keys=('self',)))

        # Get the input attributes
        self.dvh_type = dvh_type
        self.number_of_points = number_of_points
        self.reference_volumes = reference_volumes
        self.reference_doses = reference_doses

    def to_dict(self):
        """
        Serialize the evaluation handler into a dictionary.

        Returns
        -------
        dict
            Dictionary with the evaluation handler's arguments.
        """

        return vars(self)

    @classmethod
    def from_dict(
            cls,
            dictionary):
        """
        Deserialize the evaluation handler from a dictionary.

        Parameters
        ----------
        dictionary : dict
            Dictionary with the evaluation handler's arguments.

        Returns
        -------
        object of class :class:`~pyanno4rt.base._evaluation.Evaluation`
            The object used to handle the plan evaluation parameters.
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
            'dvh_type': (
                partial(validate_type, options=str),
                partial(validate_item_in_set, options=(
                    'cumulative', 'differential'))
                ),
            'number_of_points': (
                partial(validate_type, options=int),
                partial(validate_item, reference=1, sign='>=')
                ),
            'reference_volumes': (
                partial(validate_type, options=(tuple, list)),
                partial(validate_subtype, options=(int, float)),
                partial(validate_item, reference=0, sign='>='),
                partial(validate_item, reference=100, sign='<=')
                ),
            'reference_doses': (
                partial(validate_type, options=(tuple, list)),
                partial(validate_subtype, options=(int, float)),
                partial(validate_item, reference=0, sign='>=')
                )
            }

        # Loop over the inputs
        for key, value in inputs.items():

            # Loop over the validation functions
            for function in validation_map[key]:

                # Run the validation function
                function(key, value)
