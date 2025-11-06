"""Evaluation handler."""

# Author: Tim Ortkamp

# %% External package import

from functools import partial

# %% Internal package import

from pyanno4rt.tools import filter_dict
from pyanno4rt.validation import (
    validate_subtype, validate_type, validate_value, validate_value_in_set)

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

    reference_volume : list, default=[2, 5, 50, 95, 98]
        Reference volumes for the inverse DVH values.

    reference_dose : list, default=[]
        Reference doses for the DVH values.

        .. note:: If the default value is used, reference dose levels will be \
            determined automatically.

    display_segments : list, default=[]
        Names of the segments to be displayed.

        .. note:: If the default value is used, only the segments associated \
            with optimization components will be displayed.

    display_metrics : list, default=[]
        Names of the evaluation metrics to be displayed.

        .. note:: If the default value is used, all metrics will be displayed.

            Currently available:

            - 'mean': mean dose
            - 'std': standard deviation of the dose
            - 'max': maximum dose
            - 'min': minimum dose
            - 'Dx': dose quantile(s) for level x (~reference_volume)
            - 'Vx': volume quantile(s) for level x (~reference_dose)
            - 'CI': conformity index
            - 'HI': homogeneity index

    Attributes
    ----------
    dvh_type : {'cumulative', 'differential'}
        See 'Parameters'.

    number_of_points : int
        See 'Parameters'.

    reference_volume : list
        See 'Parameters'.

    reference_dose : list
        See 'Parameters'.

    display_segments : list
        See 'Parameters'.

    display_metrics : list
        See 'Parameters'.
    """

    def __init__(
            self,
            dvh_type='cumulative',
            number_of_points=1000,
            reference_volume=None,
            reference_dose=None,
            display_segments=None,
            display_metrics=None):

        # Get the input arguments
        inputs = filter_dict(vars(), remove_keys=('self',))

        # Loop over the keys with mutable default values
        for key, default in {
                'reference_volume': [2, 5, 50, 95, 98],
                'reference_dose': [],
                'display_segments': [],
                'display_metrics': []
                }.items():

            # Update the input argument value
            inputs[key] = inputs.get(key) or default

        # Validate the input arguments
        self.validate(inputs)

        # Loop over the input arguments
        for key, value in inputs.items():

            # Set the attribute
            setattr(self, key, value)

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
            Dictionary with the evaluation parameters.

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
                partial(validate_value_in_set, options=(
                    'cumulative', 'differential'))),
            'number_of_points': (
                partial(validate_type, options=int),
                partial(validate_value, reference=1, sign='>=')),
            'reference_volume': (
                partial(validate_type, options=list),
                partial(validate_subtype, options=(int, float)),
                partial(validate_value, reference=0, sign='>='),
                partial(validate_value, reference=100, sign='<=')
                ),
            'reference_dose': (
                partial(validate_type, options=list),
                partial(validate_subtype, options=(int, float)),
                partial(validate_value, reference=0, sign='>=')),
            'display_segments': (
                partial(validate_type, options=list),
                partial(validate_subtype, options=str)),
            'display_metrics': (
                partial(validate_type, options=list),
                partial(validate_subtype, options=str),
                partial(validate_value_in_set, options=(
                    'mean', 'std', 'max', 'min', 'Dx', 'Vx', 'CI', 'HI')))}

        # Loop over the inputs
        for key, value in inputs.items():

            # Loop over the validation functions
            for function in validation_map[key]:

                # Run the validation function
                function(key, value)
