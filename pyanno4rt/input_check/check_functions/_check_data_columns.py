"""Data columns checking."""

# Author: Tim Ortkamp

# %% Function definition


def check_data_columns(label, data, columns, segments, check_functions):
    """
    Check the model data columns.

    Parameters
    ----------
    label : str
        Label for the item to be checked ('data_columns').

    data : dict
        Dictionary with the user-defined data columns.

    columns : tuple
        Tuple with the data columns loaded from the file path.

    segments : dict
        Dictionary with the segment names and types.

    check_functions : tuple
        Tuple with the individual check functions for the dictionary items.
    """

    def check_single_column(path, value):
        """Check a single data column."""

        # Check if 'type' is an unavailable key
        check_functions[2](path, value)

        # Check if the column type is not a string
        check_functions[3](f"{path}['type']", value['type'])

        # Check if the column type is neither 'feature' nor 'label'
        check_functions[4](f"{path}['type']", value['type'])

        # Check if the column type is 'feature'
        if value['type'] == 'feature':

            # Check if any required key is unavailable
            check_functions[5](path, value)

            # Check if the column scale is not a string
            check_functions[6](f"{path}['scale']", value['scale'])

            # Check if the column scale is invalid
            check_functions[7](f"{path}['scale']", value['scale'])

            # Check if the column value has an invalid type
            check_functions[8](f"{path}['value']", value['value'])

            # Check if the column function is not a string
            check_functions[9](f"{path}['function']", value['function'])

            # Check if the column function is invalid
            check_functions[10](f"{path}['function']", value['function'])

            # Check if the column function has an argument
            if value['function'] in (
                    'Dx', 'Vx', 'Dose Gradient', 'Dose Moment',
                    'Dose Subvolume'):

                # Check if 'argument' is an unavailable key
                check_functions[11](path, value)

                # Check if the column function is 'Dx'
                if value['function'] == 'Dx':

                    # Check if the column argument is not numeric
                    check_functions[12](
                        f"{path}['argument']", value['argument'])

                    # Check if the column argument is > 0
                    check_functions[13](
                        f"{path}['argument']", value['argument'])

                    # Check if the column argument is < 100
                    check_functions[14](
                        f"{path}['argument']", value['argument'])

                # Check if the column function is 'Vx'
                if value['function'] == 'Vx':

                    # Check if the column argument is not numeric
                    check_functions[15](
                        f"{path}['argument']", value['argument'])

                    # Check if the column argument is > 0
                    check_functions[16](
                        f"{path}['argument']", value['argument'])

                    # Check if the column argument is < 100
                    check_functions[17](
                        f"{path}['argument']", value['argument'])

                # Check if the column function is 'Dose Gradient'
                if value['function'] == 'Dose Gradient':

                    # Check if the column argument is not a string
                    check_functions[18](
                        f"{path}['argument']", value['argument'])

                    # Check if the column argument is invalid
                    check_functions[19](
                        f"{path}['argument']", value['argument'])

                # Check if the column function is 'Dose Moment'
                if value['function'] == 'Dose Moment':

                    # Check if the column argument is not a list
                    check_functions[20](
                        f"{path}['argument']", value['argument'])

                    # Check if any column argument element is not numeric
                    check_functions[21](
                        f"{path}['argument']", value['argument'])

                    # Check if the column argument is >= 1
                    check_functions[22](
                        f"{path}['argument']", value['argument'])

                # Check if the column function is 'Dose Subvolume'
                if value['function'] == 'Dose Subvolume':

                    # Check if the column argument is not a string
                    check_functions[23](
                        f"{path}['argument']", value['argument'])

                    # Check if the column argument is invalid
                    check_functions[24](
                        f"{path}['argument']", value['argument'])

            # Check if the column segment is not a string
            check_functions[25](f"{path}['segment']", value['segment'])

            # Check if the column segment is invalid
            check_functions[26](
                f"{path}['segment']", value['segment'], segments)

        else:

            # Check if any required key is unavailable
            check_functions[27](path, value)

            # Check if the column viewpoint is not a string
            check_functions[28](f"{path}['viewpoint']", value['viewpoint'])

            # Check if the column viewpoint is invalid
            check_functions[29](f"{path}['viewpoint']", value['viewpoint'])

            # Check if the column time variable is not a string or None
            check_functions[30](
                f"{path}['time_variable']", value['time_variable'])

            # Check if the column time variable is not None
            if value['time_variable'] is not None:

                # Check if the column time variable is invalid
                check_functions[31](
                    f"{path}['time_variable']", value['time_variable'],
                    columns)

            # Check if the column bounds are not a list
            check_functions[32](f"{path}['bounds']", value['bounds'])

            # Check if any column bounds element is numeric or None
            check_functions[33](f"{path}['bounds']", value['bounds'])

    # Get the data column types
    column_types = tuple(map(lambda value: value['type'], data.values()))

    # Check if no column types have been passed
    if len(column_types) == 0:

        # Raise an error to indicate missing data columns
        raise IndexError(
            f"The treatment plan parameter 'model_parameters['{label}']' does "
            "not contain any items. Please define the feature(s) and a label!")

    # Check if no features have been passed
    if column_types.count('feature') == 0:

        # Raise an error to indicate missing features
        raise ValueError(
            f"The treatment plan parameter 'model_parameters['{label}']' does "
            "not contain any item of type 'feature'!")

    # Check if more or less than one label has been passed
    if column_types.count('label') != 1:

        # Raise an error to indicate a missing label
        raise ValueError(
            f"The treatment plan parameter 'model_parameters['{label}']' must "
            "have exactly one item of type 'label' (unsupervised or "
            "multi-label learning is not yet supported)!")

    # Loop over the dictionary keys
    for key in data:

        # Check the dictionary key
        check_functions[0](label, key, columns)

        # Get the value for the key
        value = data[key]

        # Get the dictionary path to check
        path = f"model_parameters['{label}']['{key}']"

        # Check if the value is not a dictionary
        check_functions[1](path, value)

        # Check the column
        check_single_column(path, value)
