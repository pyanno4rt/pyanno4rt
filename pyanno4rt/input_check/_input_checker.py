"""Input checker."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% Internal package import

from pyanno4rt.input_check.check_maps import (
    configuration_map, evaluation_map, model_map, objective_map,
    optimization_map, top_level_map)
from pyanno4rt.tools import flatten

# %% Class definition


class InputChecker():
    """
    Input checker class.

    """

    def __init__(self):

        # Get the checking maps
        check_maps = (top_level_map, configuration_map, optimization_map,
                      evaluation_map, objective_map, model_map)

        # Get the keys of the combined dictionaries
        all_keys = tuple(flatten(
            [dictionary.keys() for dictionary in check_maps]))

        # Check if there are duplicate keys
        if len(all_keys) != len(set(all_keys)):

            # Raise an error to indicate non-unique keys
            raise ValueError(
                "The dictionaries to be checked should only contain unique "
                "keys, but it seems that there are duplicates!")

        # Build the full check map
        self.check_map = {key: value
                          for dictionary in check_maps
                          for key, value in dictionary.items()}

    def approve(
            self,
            input_dictionary):
        """
        Approve the input dictionary by running the check functions.

        Parameters
        ----------
        input_dictionary : dict
            The dictionary whose items should be checked.

        Raises
        ------
        TypeError
            ...

        ValueError
            ...
        """
        # Loop over the dictionary keys
        for key in input_dictionary:

            # Get the check functions from the check map
            check_functions = self.check_map[key]

            # Get the value of the key
            value = input_dictionary[key]

            # Check if the value is mandatory but missing
            if value is None and key not in ('target_imaging_resolution',
                                             'initial_fluence_vector',
                                             'lower_variable_bounds',
                                             'upper_variable_bounds',
                                             'model_parameters',
                                             'model_folder_path',
                                             'link',
                                             'identifier'):

                # Raise an error to indicate the missing value
                raise TypeError(
                    "Please specify the treatment plan parameter '{}'!"
                    .format(key))

            # Loop over the check functions
            for function in check_functions:

                # Check if the key is 'solver'
                if key == 'solver':

                    # Run the check function w.r.t the optimization method
                    function(key, value,
                             value_group=input_dictionary['method'])

                # Check if the key is 'algorithm'
                elif key == 'algorithm':

                    # Run the check function w.r.t the solver
                    function(key, value,
                             value_group=input_dictionary['solver'])

                # Check if the key is 'initial_fluence_vector'
                elif key == 'initial_fluence_vector':

                    # Run the check function w.r.t the data type
                    function(key, value,
                             type_group=input_dictionary['initial_strategy'])

                # Check if the key is 'lower_variable_bounds' or
                # 'upper_variable_bounds'
                elif key in ('lower_variable_bounds', 'upper_variable_bounds'):

                    # Run the check function w.r.t the value dimension
                    function(key, value, value_group='scalar'
                             if isinstance(input_dictionary[key], (int, float,
                                                                   type(None)))
                             else 'vector')

                else:

                    # Run the check function without conditions
                    function(key, value)
