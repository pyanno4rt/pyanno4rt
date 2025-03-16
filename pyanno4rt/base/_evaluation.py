"""Plan evaluation information."""

# Author: Tim Ortkamp

# %% External package import

from functools import partial

# %% Internal package import

from pyanno4rt.input_check.check_functions import (
    check_subtype, check_type, check_value, check_value_in_set)
from pyanno4rt.tools import filter_dict

# %% Class definition


class Evaluation():
    """
    Plan evaluation information class.

    This class provides methods to set, validate and serialize the evaluation \
    parameters of the treatment plan.

    Parameters
    ----------
    dvh_type : {'cumulative', 'differential'}, default=cumulative'
        Type of DVH to be evaluated.

    number_of_points : int, default=1000
        Number of (evenly-spaced) points for which to evaluate the DVH.

    reference_volume : list, default=[2, 5, 50, 95, 98]
        Reference volumes for which to evaluate the inverse DVH values.

    reference_dose : list, default=[]
        Reference dose values for which to evaluate the DVH values.

        .. note:: If the default value is used, reference dose \
            levels will be determined automatically.

    display_segments : list, default=[]
        Names of the segmented structures to be displayed.

        .. note:: If the default value is used, all segments will \
            be displayed.

    display_metrics : list, default=[]
        Names of the plan evaluation metrics to be displayed.

        .. note:: If the default value is used, all metrics will be \
            displayed.

            The following metrics are currently available:

            - 'mean': mean dose
            - 'std': standard deviation of the dose
            - 'max': maximum dose
            - 'min': minimum dose
            - 'Dx': dose quantile(s) for level x (reference_volume)
            - 'Vx': volume quantile(s) for level x (reference_dose)
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

        # Check the input arguments
        self.check(filter_dict(vars(), remove_keys=('self',)))

        # Loop over the input arguments
        for key, value in filter_dict(vars(), remove_keys=('self',)).items():

            # Set the attribute
            setattr(self, key, value)

    def to_dict(self):
        """Return the attribute dictionary."""

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
                partial(check_value, reference=0, sign='>')),
            'display_metrics': (
                partial(check_type, types=list),
                partial(check_subtype, types=str),
                partial(check_value_in_set, options=(
                    'mean', 'std', 'max', 'min', 'Dx', 'Vx', 'CI', 'HI'))),
            'display_segments': (
                partial(check_type, types=list),
                partial(check_subtype, types=str))}

        # Loop over the dictionary items
        for key, value in input_dictionary.items():

            # Loop over the check functions
            for function in check_map[key]:

                # Run the check function
                function(key, value)
