"""Tabular dataset."""

# Author: Tim Ortkamp

# %% External package import

from math import inf
from os.path import splitext

from functools import partial
from itertools import compress, tee
from numpy import array, logical_and, seterr, vstack, where

# %% Internal package import

from pyanno4rt.io.model_data import CSVHandler
from pyanno4rt.learning._maps import FEATURES
from pyanno4rt.logging import get_logger
from pyanno4rt.tools import custom_round, deduplicate, filter_dict, replace_nan
from pyanno4rt.validation import validate_file, validate_length, validate_type

# %% Set package options

seterr(divide='ignore', invalid='ignore')

# %% Class definition


class TabularDataset():
    """
    Tabular dataset class.

    This class provides methods to handle a tabular base dataset.

    Parameters
    ----------
    path : str
        Path to the dataset.

    columns : list
        The objects used to represent the relevant dataset columns.

        Currently available:

        - :class:`~pyanno4rt.learning.features._columns.DynamicFeature`

        - :class:`~pyanno4rt.learning.features._columns.Label`

        - :class:`~pyanno4rt.learning.features._columns.StaticFeature`

        .. note:: This list acts as a filter on the raw dataset, i.e., after \
            loading the external file, only columns included will be retained.

    Attributes
    ----------
    sources : None or dict
        Dictionary with information on the external file sources and handlers.

    path : str
        See 'Parameters'.

    columns : list
        See 'Parameters'.

    dataframe : object of class :class:`~pandas.core.frame.DataFrame`
        A pandas dataframe for the dataset.

    feature_names : tuple
        Feature names.

    feature_values : ndarray
        Feature values.

    feature_scales : tuple
        Feature scaling.

    label_name : str
        Label name.

    label_values : ndarray
        Label values.

    label_bounds : list
        Label bounds for binarization.

    label_viewpoint : {'early', 'late', 'longitudinal', 'long-term'}
        Label viewpoint for temporal modulation.

    time_variable_name : None or str
        Time variable name for temporal modulation.

    time_variable_values : ndarray
        Time variable values.

    feature_map : dict
        Dictionary with mappings between features and calculation functions.
    """

    # Map the path extensions to the handlers
    sources = {
        '.csv': ('CSV file', CSVHandler)}

    def __init__(
            self,
            path,
            columns):

        # Get the input arguments
        self.inputs = filter_dict(vars(), remove_keys=('self',))

        # Check the input arguments
        self.validate(self.inputs)

        # Log a message about the initialization of the class
        get_logger().info("Initializing tabular dataset ...")

        # Get the instance attributes
        self.path = path
        self.columns = columns

        # Initialize the data attributes
        self.dataframe = None
        self.feature_names = None
        self.feature_values = None
        self.feature_scales = None
        self.label_name = None
        self.label_values = None
        self.label_bounds = None
        self.label_viewpoint = None
        self.time_variable_name = None
        self.time_variable_values = None

        # Initialize the feature map
        self.feature_map = None

    def load(self):
        """Load the dataset."""

        # Get the file string and handler
        source, handler = self.sources[splitext(self.path)[1]]

        # Log a message about the dataset loading
        get_logger().info("Importing dataset from %s ...", source)

        # Load the dataset
        self.dataframe = handler().load(self.path)[
            [item.column for item in self.columns]]

    def save(
            self,
            path):
        """
        Save the dataset.

        Parameters
        ----------
        path : str
            Path for storing the dataset.
        """

        # Get the file string and handler
        source, handler = self.sources[splitext(path)[1]]

        # Log a message about the dataset saving
        get_logger().info("Saving dataset to %s ...", source)

        # Save the dataset
        handler().save(self.dataframe, path)

    def generate(self):
        """Generate the data attributes."""

        # Load the dataset
        self.load()

        # Decompose the dataset
        self._decompose()

        # Modulate the dataset
        self._modulate()

        # Binarize the label values
        self._binarize()

        # Build the feature map
        self._build_map()

    def _decompose(self):
        """Decompose the base tabular dataset."""

        # Log a message about the dataset decomposition
        get_logger().info(
            "Decomposing tabular dataset into features, label and time "
            "variable ...")

        # Get the feature objects
        features = tuple(
            column for column in self.columns if column.category == 'feature')

        # Set the feature names and scales
        self.feature_names, self.feature_scales = zip(*[
            (feature.column, feature.scale) for feature in features])

        # Set the feature values
        self.feature_values = self.dataframe.drop(
            self.dataframe.columns.difference(self.feature_names),
            axis=1).values

        # Get the label object
        label = next(iter(
            column for column in self.columns if column.category == 'label'))

        # Set the label name, viewpoint and bounds
        self.label_name, self.label_viewpoint, self.label_bounds = (
            label.column, label.viewpoint, label.bounds)

        # Set the label values
        self.label_values = self.dataframe[self.label_name].values

        # Set the time variable name
        self.time_variable_name = label.time_variable

        # Set the time variable values
        self.time_variable_values = self.dataframe[
            filter(None, [self.time_variable_name])].values

    def _modulate(self):
        """Modulate the data information."""

        def squeeze_labels(bounds, index_sets):
            """Squeeze the labels per patient by the time bounds."""

            # Get a boolean mask indicating interior samples per patient
            interior_mask = tee((
                logical_and(
                    bounds[0]*365/12 <= self.time_variable_values[indices],
                    bounds[1]*365/12 > self.time_variable_values[indices]
                    ).reshape(-1) for indices in index_sets), 2)

            # Get the label values from the interior samples per patient
            interior_labels = (
                self.label_values[list(compress(value[0], value[1]))]
                for value in zip(index_sets, interior_mask[0]))

            # Get the mean interior label value per patient
            interior_means = replace_nan((
                numerator/denominator for numerator, denominator in zip(
                    map(sum, interior_labels), map(sum, interior_mask[1]))),
                0.0)

            return array(list(map(custom_round, interior_means)))

        # Map the label viewpoints to the time bounds
        viewpoints = {
            'early': ((0,), (6,)),
            'late': ((6,), (15,)),
            'long-term': ((15,), (24,)),
            'longitudinal': ((), ())}

        # Check if the label viewpoint is not 'longitudinal'
        if self.label_viewpoint != 'longitudinal':

            # Log a message about the dataset modulation
            get_logger().info(
                "Modulating dataset by feature '%s' for label viewpoint '%s' "
                "...", self.time_variable_name, self.label_viewpoint)

            # Get the mapping between patient features and sample indices
            patient_map = deduplicate(map(tuple, self.feature_values))

            # Overwrite the feature values by the patient features
            self.feature_values = array((*patient_map,))

            # Overwrite the label values by the squeezed labels
            self.label_values = vstack(tuple(map(
                partial(squeeze_labels, index_sets=patient_map.values()),
                zip(*viewpoints[self.label_viewpoint])))).T.reshape(-1)

    def _binarize(self):
        """Binarize the label values."""

        # Transform the label bounds by replacing None with limit values
        label_bounds = [
            self.label_bounds[index] if self.label_bounds[index] is not None
            else (-1)**(index+1)*inf for index in range(2)]

        # Log a message about the dataset binarization
        get_logger().info(
            "Binarizing data information by label bounds %s ...", label_bounds)

        # Overwrite the label values with the binarizations
        self.label_values = where(
            (self.label_values >= label_bounds[0])
            & (self.label_values <= label_bounds[1]), 1, 0)

    def _build_map(self):
        """Build the feature map."""

        def get_single_map(feature):
            """Get the mapping for a single feature."""

            # Check if the feature has a fixed value
            if hasattr(feature, 'function'):

                # Get the calculation function
                function = FEATURES[feature.function]

                # Get the function argument
                args = feature.argument

                # Return the dosiomic/radiomic feature map
                return {
                    'name': feature.function,
                    'segment': feature.segment,
                    'class': function.feature_class,
                    'computation': (
                        methods[args is None](function.compute, args)),
                    'differentiation': (
                        methods[args is None](function.differentiate, args)
                        if function.feature_class == 'Dosiomics' else None)}

            # Return the static feature map
            return {
                'value': feature.value,
                'segment': None,
                'class': 'Statics',
                'computation': None,
                'differentiation': None}

        # Log a message about the feature map build
        get_logger().info("Building the feature map ...")

        # Create a boolean mapping to the internal functions
        methods = {True: (lambda x, y: x), False: partial}

        # Get the feature columns
        features = [
            item for item in self.columns if item.category == 'feature']

        # Set the feature map
        self.feature_map = {
            feature.column: get_single_map(feature) for feature in features}

    def validate(
            self,
            inputs):
        """
        Validate the input arguments.

        Parameters
        ----------
        inputs : dict
            Dictionary with the mappings between argument names and values.
        """

        validation_map = {
            'path': (
                partial(validate_type, options=str),
                partial(validate_file, options=('.csv',))
                ),
            'columns': (
                partial(validate_type, options=list),
                partial(validate_length, reference=2, sign='>=')
                )
            }

        # Loop over the dictionary items
        for key, value in inputs.items():

            # Loop over the validation functions
            for function in validation_map[key]:

                # Run the validation function
                function(key, value)

            # Check if the key is 'column'
            if key == 'columns':

                # Check if no feature has been passed
                if sum(item.category == 'feature'
                       for item in inputs['columns']) == 0:

                    # Raise an error to indicate missing features
                    raise ValueError(
                        "The tabular dataset parameter 'columns' does not "
                        "include at least one item of type 'DynamicFeature' "
                        "or 'StaticFeature'!")

                # Check if not exactly one label has been passed
                if sum(item.category == 'label'
                       for item in inputs['columns']) != 1:

                    # Raise an error to indicate a non-unique label
                    raise ValueError(
                        "The tabular dataset parameter 'columns' does not "
                        "include exactly one item of type 'Label'!")
