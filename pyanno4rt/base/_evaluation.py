"""Evaluation handler."""

# Author: Tim Ortkamp

# %% External package import

from functools import partial

# %% Internal package import

from pyanno4rt.checking import (
    check_subtype, check_type, check_value, check_value_in_set)
from pyanno4rt.tools import filter_dict

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
        Reference volumes for which to evaluate the inverse DVH values.

    reference_dose : list, default=[]
        Reference dose values for which to evaluate the DVH values.

        .. note:: If the default value is used, reference dose levels will be \
            determined automatically.

    display_segments : list, default=[]
        Names of the segments to be displayed.

        .. note:: If the default value is used, all segments will be displayed.

    display_metrics : list, default=[]
        Names of the evaluation metrics to be displayed.

        .. note:: If the default value is used, all metrics will be displayed.

            Currently available:

            - 'mean': mean dose
            - 'std': standard deviation of the dose
            - 'max': maximum dose
            - 'min': minimum dose
            - 'Dx': dose quantile(s) for level x (-> reference_volume)
            - 'Vx': volume quantile(s) for level x (-> reference_dose)
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
            reference_volume=[2, 5, 50, 95, 98],
            reference_dose=[],
            display_segments=[],
            display_metrics=[]):

        # Get the input arguments
        inputs = filter_dict(vars(), remove_keys=('self',))

        # Check the input arguments
        self.check(inputs)

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
            'reference_volume': (
                partial(check_type, types=list),
                partial(check_subtype, types=(int, float)),
                partial(check_value, reference=0, sign='>=', is_vector=True),
                partial(check_value, reference=100, sign='<=', is_vector=True)
                ),
            'reference_dose': (
                partial(check_type, types=list),
                partial(check_subtype, types=(int, float)),
                partial(check_value, reference=0, sign='>=', is_vector=True)),
            'dvh_type': (
                partial(check_type, types=str),
                partial(check_value_in_set, options=(
                    'cumulative', 'differential'))),
            'number_of_points': (
                partial(check_type, types=int),
                partial(check_value, reference=1, sign='>=')),
            'display_metrics': (
                partial(check_type, types=list),
                partial(check_subtype, types=str),
                partial(check_value_in_set, options=(
                    'mean', 'std', 'max', 'min', 'Dx', 'Vx', 'CI', 'HI'))),
            'display_segments': (
                partial(check_type, types=list),
                partial(check_subtype, types=str))}

        # Loop over the inputs
        for key, value in inputs.items():

            # Loop over the check functions
            for function in check_map[key]:

                # Run the check function
                function(key, value)
