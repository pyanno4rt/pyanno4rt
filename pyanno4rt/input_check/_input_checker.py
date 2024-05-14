"""Input checker."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from warnings import warn

# %% Internal package import

from pyanno4rt.input_check.check_maps import (
    component_map, configuration_map, evaluation_map, model_display_map,
    model_map, optimization_map, top_level_map, tune_space_map)
from pyanno4rt.tools import flatten

# %% Class definition


class InputChecker():
    """
    Input checker class.

    This class provides methods to perform input checks on the user-defined \
    parameters for objects of any class from :mod:`~pyanno4rt.base`. It \
    ensures the validity of the internal program steps with regard to the \
    exogenous variables.

    Attributes
    ----------
    check_map : dict
        Dictionary with all mappings between parameter names and validity \
        check functions.

    Raises
    ------
    ValueError
        If non-unique parameter names are found.

    Notes
    -----
    The :class:`~pyanno4rt.input_check._input_checker.InputChecker` class \
    relies on the uniqueness of the parameter names to create a \
    dictionary-based mapping. Hence, make sure to assign unique labels for \
    all parameters to be checked!
    """

    def __init__(self):

        # Get all available check maps
        check_maps = (component_map, configuration_map, evaluation_map,
                      model_map, model_display_map, optimization_map,
                      top_level_map, tune_space_map)

        # Get all parameter names
        parameter_names = tuple(flatten([
            dictionary.keys() for dictionary in check_maps]))

        # Check if there are duplicate keys
        if len(parameter_names) != len(set(parameter_names)):

            # Raise an error to indicate non-unique keys
            raise ValueError(
                "The check maps should only contain unique keys, but it seems "
                "that there are duplicates within or between some maps!")

        # Build the full check map
        self.check_map = {key: value
                          for dictionary in check_maps
                          for key, value in dictionary.items()}

    def approve(
            self,
            input_dictionary):
        """
        Approve the input dictionary items (parameter names and values) by \
        running the corresponding check functions.

        Parameters
        ----------
        input_dictionary : dict
            Dictionary with the mappings between parameter names and values \
            to be checked.
        """

        # Set the additional check function arguments
        args = {'solver': {
                    'value_condition': input_dictionary.get('method')},
                'algorithm': {
                    'value_condition': (
                        f"{input_dictionary.get('method')}/"
                        f"{input_dictionary.get('solver')}")},
                'initial_fluence_vector': {
                    'type_condition': input_dictionary.get(
                        'initial_strategy')},
                'time_variable_name': {
                    'type_condition': input_dictionary.get('label_viewpoint')}
                }

        # Loop over the dictionary keys
        for key, value in input_dictionary.items():

            # Check if the key is included in the check map
            if key in self.check_map:

                # Check if the key holds vector-like lower or upper bounds
                if (key in ('lower_variable_bounds', 'upper_variable_bounds')
                        and not isinstance(value, (int, float, type(None)))):

                    # Add the corresponding additional argument
                    args[key] = {'is_vector': True}

                # Loop over the check functions
                for function in self.check_map[key]:

                    # Get the additional arguments
                    key_args = args.get(key, {})

                    # Get the function arguments
                    func_args = function.func.__code__.co_varnames

                    # Get the additional arguments filtered by function
                    filter_args = {arg: key_args[arg] for arg in func_args
                                   if arg in key_args}

                    # Run the check function
                    function(key, value, **filter_args)

            else:

                # Raise a warning to indicate an uncheckable parameter
                warn(f"The key '{key}' cannot be found in the check map and "
                     "is therefore not approved!")
