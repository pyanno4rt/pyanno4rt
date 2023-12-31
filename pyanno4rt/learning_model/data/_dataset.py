"""Dataset handling & modulation."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from itertools import compress
from math import isnan
from numpy import array, empty, vstack, where
from pandas import read_csv

# %% Internal package import

from pyanno4rt.datahub import Datahub

# %% Class definition


class Dataset():
    """
    Dataset handling class.

    This class provides methods to load, deconstruct and modulate the base \
    data set according to the selected features and label viewpoint.

    Parameters
    ----------
    data_path : string
        Path to the data set used for fitting the learning models, needs to \
        be a .csv file.

    feature_filter : tuple
        A feature (sub)set as an iterable and a value from {'retain', \
        'remove'} as an indicator for retaining or removing the (sub)set \
        prior to the modeling process, all features are retained if no value \
        is passed.

    label_viewpoint : {'early', 'late', 'long-term', 'longitudinal', 'profile'}
        Time of observation for the presence of tumor control and/or normal \
        tissue complications. The values can be described as follows:

        - 'early' : observation period between 0 and 6 months after treatment;
        - 'late' : observation period between 6 and 15 months after treatment;
        - 'long-term' : observation period between 15 and 24 months after \
           treatment;
        - 'longitudinal' : no observation period, time after treatment is \
           included as a covariate;
        - 'profile' : TCP/NTCP profiling over time, multi-label scenario \
           with one label per month (up to 24 labels in total).

        Modification of the time interval bounds can be done in the \
        'viewpoints' dictionary at the end of the file.

    label_bounds : list
        Bounds for the label values to binarize them into positive and \
        negative class, i.e., label values within the specified bounds will \
        be interpreted as a binary 1.

    Attributes
    ----------
    data_path : string
        See 'Parameters'.

    feature_filter : tuple
        See 'Parameters'.

    label_viewpoint : {'early', 'late', 'long-term', 'longitudinal', 'profile'}
        See 'Parameters'.

    label_bounds : list
        See 'Parameters'.

    dataset : dict
        Dictionary with the raw data set, the label viewpoint, the feature \
        values and names, and the label values and names after modulation. In \
        a compact way, this represents the input data for the learning models.
    """

    def __init__(
            self,
            model_label,
            data_path,
            feature_filter,
            label_viewpoint,
            label_bounds):

        # Initialize the datahub
        hub = Datahub()

        # Log a message about the dataset building
        hub.logger.display_info("Building base dataset ...")

        # Get the instance attributes from the arguments
        self.model_label = model_label
        self.data_path = data_path
        self.feature_filter = feature_filter
        self.label_viewpoint = label_viewpoint
        self.label_bounds = label_bounds

        # Generate the data set
        hub.datasets[model_label] = self.generate_dataset()

    def generate_dataset(self):
        """
        Deconstruct the raw dataset and return a dictionary with the \
        relevant components of it.

        Returns
        -------
        dict
            Dictionary with all necessary information about the data set, \
            i.e., raw data, label viewpoint, feature values and names, and \
            label values and names.
        """
        def get_feature_info(data):
            """Get the feature values and names from the data set."""
            # Get the feature values from all but the last column
            feature_values = data.iloc[:, :-1]

            # Get the feature names as a list
            feature_names = list(feature_values.columns)

            return array(feature_values), feature_names

        def get_label_info(data):
            """Get the label values and names from the data set."""
            # Get the label values from the last column
            label_values = data.iloc[:, -1]

            # Get the label name as a list
            label_names = [data.columns[-1]]

            return array(label_values), label_names

        # Read the data from the .csv file
        data = read_csv(self.data_path)

        # Select/Drop the features according to the filtering list
        if self.feature_filter[1] == 'retain':
            data = self.retain_features(data)
        else:
            data = self.remove_features(data)

        # Extract the feature and label information
        feature_values, feature_names = get_feature_info(data)
        label_values, label_names = get_label_info(data)

        # Modulate the data set
        if 'patientDaysafterrt' in feature_names:
            feature_values, feature_names, label_values = self.modulate_data(
                feature_values, feature_names, label_values)

        # Define the output dictionary keys and values
        keys = ('raw_data', 'label_viewpoint', 'label_bounds',
                'feature_values', 'feature_names', 'label_values',
                'label_names')
        values = (data, self.label_viewpoint, self.label_bounds,
                  feature_values, feature_names, label_values, label_names)

        return dict(zip(keys, values))

    def retain_features(
            self,
            data):
        """
        Retain the features in the filter set from the data set.

        Parameters
        ----------
        data : DataFrame
            Two-dimensional tabular data structure with the feature and label \
            information.

        Returns
        -------
        DataFrame
            Filtered tabular data with the features to be retained.
        """
        return data.drop(data.columns.difference(
            self.feature_filter[0]), axis=1, inplace=False)

    def remove_features(
            self,
            data):
        """
        Remove the features in the filter set from the data set.

        Parameters
        ----------
        data : DataFrame
            Two-dimensional tabular data structure with the feature and label \
            information.

        Returns
        -------
        DataFrame
            Filtered tabular data without the features to be removed.
        """
        return data.drop(self.feature_filter[0], axis=1, inplace=False)

    def modulate_data(
            self,
            feature_values,
            feature_names,
            label_values):
        """
        Modulate the data set depending on the label viewpoint.

        Parameters
        ----------
        feature_values : ndarray
            Values of the features.

        feature_names : list
            Names of the features.

        label_values : ndarray
            Values of the labels.

        Returns
        -------
        tuple
            Tuple with the modulated feature values, feature names and label \
            values.
        """
        # Log a message about the dataset modulation
        Datahub().logger.display_info("Modulating dataset for label viewpoint "
                                      "'{}' ..."
                                      .format(self.label_viewpoint))

        def round_custom(number):
            """Round numbers in a custom way."""
            # Get the number with two decimal places as a string
            number = str(number)[:str(number).index('.') + 2]

            # Round the number
            if number[-1] >= '5':
                rounded_number = float(number[:-3] + str(int(number[-3]) + 1))
            else:
                rounded_number = float(number[:-1])

            return rounded_number

        def deduplicate(iterable):
            """Convert an iterable to a dictionary with indices per unique \
                element."""
            # Initialize the mapping dictionary
            mapping = {}

            # Set the starting index to zero
            index = 0

            # Loop over all elements in the iterable
            for elem in iterable:

                # Check if the element is already a dictionary key
                if elem in mapping:

                    # Add the index to the values
                    mapping[elem] += (index,)

                else:

                    # Create a new key and initialize the index list
                    mapping[elem] = [index]

                # Increment the index
                index += 1

            # Convert the index lists (values) into tuples
            mapping = {key: tuple(value) for key, value in mapping.items()}

            return mapping

        def replace_nan(iterable, value):
            """Replace the 'nan' entries in an iterable by a specific value."""
            return (value if isnan(elem) else elem for elem in iterable)

        def modulate_by_time(lower_bounds, upper_bounds, severity,
                             feature_values=feature_values,
                             feature_names=feature_names,
                             label_values=label_values):
            """Modulate the features and labels depending on the time after \
            treatment."""
            # Collect the duplicate feature vectors (patients) in a dictionary
            vector_index_map = deduplicate(map(tuple, feature_values[:, :-1]))

            # Initialize the modified labels array
            modified_labels = empty(shape=(len(vector_index_map.keys()),))

            # Iterate over the pairs of lower and upper bounds
            for lower_bound, upper_bound in zip(lower_bounds, upper_bounds):

                try:

                    # Convert the bounds from month into day unit
                    lower_days = lower_bound * 365/12
                    upper_days = upper_bound * 365/12

                except TypeError:

                    # Return the raw features and the binarized labels
                    return feature_values, feature_names, where(
                        label_values >= severity[0]
                        & label_values <= severity[1],
                        1, 0)

                # Create a boolean list for each item in the deduplicated
                # dictionary indicating which values of the post-treatment
                # days feature fall under the label viewpoint
                within_viewpoint = tuple(
                    [lower_days <= feature_values[index, -1] < upper_days
                     for index in indices]
                    for indices in vector_index_map.values())

                # Create a tuple of arrays representing the values for 'grade'
                # associated with a logical value of 1 in 'within_viewpoint'
                grades_viewpoint = tuple(
                    label_values[list(compress(value[0], value[1]))]
                    for value in zip(vector_index_map.values(),
                                     within_viewpoint))

                # Compute the means over all arrays in 'grades_viewpoint'
                means_viewpoint = replace_nan(
                    (numerator/denominator for numerator, denominator in zip(
                        map(sum, grades_viewpoint),
                        map(sum, within_viewpoint))),
                    0.0)

                # Round the mean values
                rounded_means_viewpoint = [
                    round_custom(mean) for mean in means_viewpoint]

                # Extend the modified labels array
                modified_labels = vstack((modified_labels, where(
                    (array(rounded_means_viewpoint) >= severity[0])
                    & (array(rounded_means_viewpoint) <= severity[1]),
                    1, 0)))

            # Get the feature values
            feature_values = array((*vector_index_map,))

            # Remove the post-treatment days feature from the feature names
            feature_names.remove('patientDaysafterrt')

            # Check if the label viewpoint is not 'profile'
            if self.label_viewpoint != 'profile':

                # Return the features and the modified 1D labels array
                return feature_values, feature_names, modified_labels[
                    1:, :].T.flatten()

            # Otherwise, return the features and the modified 2D labels array
            return feature_values, feature_names, modified_labels[1:, :].T

        # Map the values of 'label_viewpoint' to the time and label bounds
        viewpoints = {'early': ((0,), (6,), self.label_bounds),
                      'late': ((6,), (15,), self.label_bounds),
                      'long-term': ((15,), (24,), self.label_bounds),
                      'longitudinal': ((None,), (None,), self.label_bounds),
                      'profile': (range(24), range(1, 25), self.label_bounds)}

        return modulate_by_time(*viewpoints[self.label_viewpoint])
