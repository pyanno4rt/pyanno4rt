"""Input checker."""

# Author: Tim Ortkamp

# %% External package import

from warnings import warn

from pandas import read_csv

# %% Internal package import

from pyanno4rt.input_check.check_maps import (
    component_map, configuration_map, evaluation_map, model_display_map,
    model_map, optimization_map, top_level_map, tune_space_map)
from pyanno4rt.tools import flatten, load_segments_from_path

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

    imaging_path : str
        

    data_path : str
        Path to the data set used for fitting the machine learning model.

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

        # Initialize the external data paths
        self.imaging_path = None
        self.data_path = None

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
                'data_path': {
                    'type_condition': isinstance(
                        input_dictionary.get('model_folder_path'), str)},
                'data_columns': {
                    'type_condition': isinstance(
                        input_dictionary.get('model_folder_path'), str)}}

        # Check if the type condition on the data path is fulfilled
        if args['data_path']['type_condition']:

            # Reduce the check map for the data path
            self.check_map['data_path'] = (self.check_map['data_path'][0],)

        # Check if the type condition on the data columns is fulfilled
        if args['data_columns']['type_condition']:

            # Reduce the check map for the data path
            self.check_map['data_columns'] = (
                self.check_map['data_columns'][0],)

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

                    # Check if the function is 'check_components'
                    if function.func.__name__ == 'check_components':

                        # Extend the arguments by the segments
                        filter_args |= {
                            'segments': load_segments_from_path(
                                self.imaging_path)}

                    # Else, check if the function is 'check_data_columns'
                    elif function.func.__name__ == 'check_data_columns':

                        # Extend the arguments by the data column names
                        filter_args |= {
                            'columns': tuple(read_csv(self.data_path)),
                            'segments': load_segments_from_path(
                                self.imaging_path)}

                    # Run the check function
                    function(key, value, **filter_args)

                    # Check if the key is 'imaging_path'
                    if key == 'imaging_path':

                        # Update the imaging path
                        self.imaging_path = value

                    # Check if the key is 'data_path'
                    if key == 'data_path':

                        # Update the data path
                        self.data_path = value

            else:

                # Raise a warning to indicate an uncheckable parameter
                warn(f"The key '{key}' cannot be found in the check map and "
                     "is therefore not approved!")
