"""Tabular dataset."""

# Author: Tim Ortkamp

# %% External package import

from math import inf
from os.path import abspath, splitext

from copy import deepcopy
from functools import partial
from itertools import compress, tee
from numpy import arange, array, logical_and, seterr, vstack, where, zeros
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

# %% Internal package import

from pyanno4rt.io.model_data import CSVHandler
from pyanno4rt.learning._maps import COLUMNS, FEATURES
from pyanno4rt.logging import get_logger
from pyanno4rt.tools import custom_round, deduplicate, filter_dict, replace_nan
from pyanno4rt.validation import (
    validate_file, validate_item, validate_length, validate_subtype,
    validate_type)

# %% Set package options

seterr(divide='ignore', invalid='ignore')

# %% Class definition


class TabularDataset():
    """
    Tabular dataset class.

    This class provides methods to handle a tabular base dataset.

    Parameters
    ----------
    data_path : None or str
        Path to the dataset.

    columns : None or list
        The objects used to represent the relevant dataset columns.

        Currently available:

        - :class:`~pyanno4rt.learning.features._columns.DynamicFeature`

        - :class:`~pyanno4rt.learning.features._columns.Label`

        - :class:`~pyanno4rt.learning.features._columns.StaticFeature`

    holdout : None, float or int, default=None
        Size of the holdout set. If 0 < holdout < 1, represents the \
        proportion of the holdout dataset. If int, represents the absolute \
        number of samples.

    splits : int, default=5
        Number of splits for cross-validation. If splits = 1, a single \
        train-validation split (80/20) is generated.

    repeats : int, default=1
        Number of repeats for cross-validation (or train-validation).

    Attributes
    ----------
    _sources : dict
        Dictionary with information on the external file sources and handlers.

    arguments : dict
        Dictionary with the input arguments (for serialization).

    data_path : None or str
        See 'Parameters'.

    columns : None or list
        See 'Parameters'.

    holdout : None, float or int
        See 'Parameters'.

    splits : int
        See 'Parameters'.

    repeats : int
        See 'Parameters'.

    dataframe : object of class :class:`~pandas.core.frame.DataFrame`
        A pandas dataframe for the dataset.

    feature_names : None or tuple
        Feature names.

    feature_values : None or ndarray
        Feature values.

    feature_scales : None or tuple
        Feature scaling.

    label_name : None or str
        Label name.

    label_values : None or ndarray
        Label values.

    label_bounds : None or list
        Label bounds for binarization.

    label_viewpoint : None or {'early', 'late', 'longitudinal', 'long-term'}
        Label viewpoint for temporal modulation.

    time_variable_name : None or str
        Time variable name for temporal modulation.

    time_variable_values : None or ndarray
        Time variable values.

    folds : None or ndarray
        Fold numbers for cross-validation.

    feature_map : None or dict
        Dictionary with mappings between features and calculation functions.

    holdout_set : None or dict
        Dictionary with the holdout data.

    Notes
    -----
    Tabular datasets can be generated in three ways:

        1. By providing a data path along with column information, a full \
            decomposition and postprocessing of the features and labels \
            is performed, including a feature-to-function mapping. This is \
            the preferred way for internal model fitting and embedding into \
            the treatment plan optimization problem.

        2. By providing only a data path, column information will be inferred \
            (the first p-1 columns in the dataset are considered "static" \
             features, and the p-th column the label). This is the preferred \
            way for internal model fitting without embedding into the \
            treatment plan optimization problem.

        3. By providing only column information, a light decomposition \
            extracting feature and label names is performed, without any \
            postprocessing except for the feature-to-function mapping. This \
            is the preferred way for embedding a pre-trained external model \
            into the treatment plan optimization problem.
    """

    # Map the path extensions to the handlers
    _sources = {
        '.csv': ('CSV file', CSVHandler)}

    def __init__(
            self,
            data_path,
            columns,
            holdout=None,
            splits=5,
            repeats=1):

        # Check if a dataset path has been provided
        if data_path is not None:

            # Convert the path into an absolute value
            data_path = abspath(data_path)

        # Get the input arguments
        self.arguments = filter_dict(vars(), remove_keys=('self',))

        # Check the input arguments
        self.validate(self.arguments)

        # Log a message about the initialization of the class
        get_logger().info("Initializing tabular dataset ...")

        # Get the instance attributes
        self.data_path = data_path
        self.columns = columns
        self.holdout = holdout
        self.splits = splits
        self.repeats = repeats

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
        self.folds = None

        # Initialize the feature map
        self.feature_map = None

        # Initialize the holdout dataset
        self.holdout_set = None

    def to_dict(self):
        """
        Serialize the dataset into a dictionary.

        Returns
        -------
        dict
            Dictionary with the dataset's arguments.
        """

        # Get the parameter dictionary
        dictionary = deepcopy(self.arguments)

        # Check if column information is available
        if dictionary['columns'] is not None:

            # Get the columns

            # Serialize the columns
            dictionary['columns'] = [item.to_dict() for item in self.columns]

        return dictionary

    @classmethod
    def from_dict(
            cls,
            dictionary):
        """
        Deserialize the dataset from a dictionary.

        Parameters
        ----------
        dictionary : dict
            Dictionary with the dataset's arguments.

        Returns
        -------
        object of class \
            :class:`~pyanno4rt.learning.datasets._tabular_dataset.TabularDataset`
            The object used to represent the dataset.
        """

        # Check if column information is available
        if dictionary['columns'] is not None:

            # Deserialize the columns
            dictionary['columns'] = [
                COLUMNS[key].from_dict(value)
                for item in dictionary['columns']
                for key, value in item.items()]

        return cls(**dictionary)

    def load(self):
        """Load the dataset."""

        # Check if a data path has been provided
        if self.data_path is not None:

            # Get the file string and handler
            source, handler = self._sources[splitext(self.data_path)[1]]

            # Log a message about loading the dataset
            get_logger().info("Importing dataset from %s ...", source)

            # Load the dataset
            self.dataframe = handler().load(self.data_path)

            # Check if no column information has been provided
            if self.columns is None:

                # Log a message about inferring the column information
                get_logger().warning(
                    "User has not provided column information, inferring from "
                    "dataset ...")

                # Infer the column information
                self.columns = self.infer_columns()

            # Map the column names to the index position in the dataframe
            order = {
                name: index
                for index, name in enumerate(self.dataframe.columns)}

            # Get the invalid columns
            invalid = tuple(
                column for column in self.columns
                if column.column not in self.dataframe.columns)

            # Check if any invalid columns have been passed
            if len(invalid) > 0:

                # Log a message about the invalid columns
                get_logger().warning(
                    "User has passed column information that is not "
                    "available in the dataset "
                    f"({', '.join((column.column for column in invalid))}), "
                    "removing corresponding elements ...")

            # Sort the column objects by their order in the dataframe
            self.columns = sorted(
                list(set(self.columns)-set(invalid)),
                key=lambda x: order.get(x.column, inf))

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
        source, handler = self._sources[splitext(path)[1]]

        # Log a message about saving the dataset
        get_logger().info("Saving dataset to %s ...", source)

        # Save the dataset
        handler().save(self.dataframe, path)

    def infer_columns(self):
        """
        Infer the column information from the dataset.

        Returns
        -------
        list
            The objects used to represent the relevant dataset columns.

        Notes
        -----
        This function should later become "smart", i.e., infer the column \
        information based on e.g. fuzzy similarity matching.

        Current assumptions:

            1. All columns except the last one represent static features with \
                the label included as the last column

            2. Static values can be approximated by the column mean.
        """

        # Get the features
        features = self.dataframe.iloc[:, :-1]

        # Get the label name
        label_name = self.dataframe.columns[-1]

        # Map each feature to an aggregation function
        mapping = {
            column: 'mean' if is_numeric_dtype(dtype)
            else lambda x: x.mode().iat[0]
            for column, dtype in features.dtypes.items()}

        # Compute the static values
        statics = features.agg(mapping)

        # Initialize the column list by the feature objects
        columns = [
            COLUMNS['Static Feature'](column, value)
            for column, value in statics.items()]

        # Append the label object
        columns.append(COLUMNS['Label'](label_name))

        return columns

    def generate(self):
        """Generate the data attributes."""

        # Decompose the dataset
        self._decompose()

        # Check if a dataframe is available
        if self.dataframe is not None:

            # Modulate the dataset
            self._modulate()

            # Binarize the label values
            self._binarize()

            # Get the folds
            self._get_folds()

            # Check if a holdout set should be created
            if self.holdout is not None:

                # Create the holdout data
                self._create_holdout()

        # Build the feature map
        self._build_map()

    def _decompose(self):
        """Decompose the dataset."""

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

        # Get the label object
        label = next(iter(
            column for column in self.columns if column.category == 'label'))

        # Set the label name, viewpoint and bounds
        self.label_name, self.label_viewpoint, self.label_bounds = (
            label.column, label.viewpoint, label.bounds)

        # Set the time variable name
        self.time_variable_name = label.time_variable

        # Check if a dataframe is available
        if self.dataframe is not None:

            # Set the feature values
            self.feature_values = self.dataframe.drop(
                self.dataframe.columns.difference(self.feature_names),
                axis=1).values

            # Set the label values
            self.label_values = self.dataframe[self.label_name].values

            # Set the time variable values
            self.time_variable_values = self.dataframe[
                filter(None, [self.time_variable_name])].values

    def _modulate(self):
        """Modulate the dataset."""

        def aggregate_labels(bounds, index_sets):
            """Aggregate the labels per patient by the time bounds."""

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
                    map(sum, interior_labels),
                    map(sum, interior_mask[1]))), 0.0)

            return array(list(map(custom_round, interior_means)))

        # Map the label viewpoints to the time intervals (in months)
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

            # Get the mapping between patients and samples
            patient_map = deduplicate(map(tuple, self.feature_values))

            # Overwrite the feature values by the patient-wise features
            self.feature_values = array((*patient_map,))

            # Overwrite the label values by the aggregated labels
            self.label_values = vstack(tuple(map(
                partial(aggregate_labels, index_sets=patient_map.values()),
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

    def _get_folds(self):
        """Get the fold numbers for cross-validation (or train-validation)."""

        # Clamp the number of splits
        clamped_n_splits = min(self.splits, sum(self.label_values))

        # Initialize the stratified k-fold cross-validator
        cross_validator = RepeatedStratifiedKFold(
            n_splits=5 if clamped_n_splits == 1 else clamped_n_splits,
            n_repeats=self.repeats, random_state=123)

        # Get the stratification splits
        splits = tuple(cross_validator.split(
            self.feature_values, self.label_values))

        # Divide the splits into chunks (for each repeat)
        chunks = [
            splits[index:index+self.splits] for index in range(
                0, clamped_n_splits*self.repeats, clamped_n_splits)]

        # Initialize the fold numbers
        folds = zeros((len(self.label_values), self.repeats))

        # Loop over the chunks
        for column, chunk in enumerate(chunks):

            # Loop over the chunk splits
            for number, (_, validation_index) in enumerate(chunk):

                # Enter the fold number for the validation set repetition
                folds[validation_index, column] = (
                    int(number) if self.splits != 1 else 1)

        # Store the fold numbers
        self.folds = folds

    def _create_holdout(self):
        """Create the holdout dataset."""

        # Get the requested holdout size
        requested = (
            self.holdout if isinstance(self.holdout, int)
            else int(self.holdout*len(self.dataframe)))

        # Get the clamped size of the holdout set
        holdout = max(
            min(len(self.dataframe)-self.splits, requested), 2, self.splits)

        # Log a message about creating the holdout dataset
        get_logger().info(
            "Creating holdout dataset with %s samples (requested: %s) ...",
            holdout, requested)

        # Get the training and holdout indices
        train_indices, holdout_indices = train_test_split(
            arange(len(self.dataframe)), test_size=holdout,
            stratify=self.dataframe[self.label_name], random_state=42)

        # Get the holdout set
        self.holdout_set = {
            'dataframe': self.dataframe.iloc[holdout_indices],
            'feature_values': self.feature_values[holdout_indices],
            'label_values': self.label_values[holdout_indices],
            'time_variable_values': self.time_variable_values[holdout_indices],
            'folds': self.folds[holdout_indices]
            }

        # Reduce the training data
        self.dataframe = self.dataframe.iloc[train_indices]
        self.feature_values = self.feature_values[train_indices]
        self.label_values = self.label_values[train_indices]
        self.time_variable_values = self.time_variable_values[train_indices]
        self.folds = self.folds[train_indices]

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
                'class': 'Statics'}

        # Log a message about building the feature map
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
            'data_path': (
                partial(validate_type, options={
                    True: str,
                    False: (type(None), str)},
                    condition=inputs['columns'] is None),
                partial(validate_file, options=('.csv',))
                ),
            'columns': (
                partial(validate_type, options={
                    True: list,
                    False: (type(None), list)},
                    condition=inputs['data_path'] is None),
                partial(validate_subtype, options=(*COLUMNS.values(),)),
                partial(validate_length, reference=2, sign='>=')
                ),
            'holdout': (
                partial(validate_type, options=(type(None), float, int)),
                ),
            'splits': (
                partial(validate_type, options=int),
                partial(validate_item, reference=1, sign='>=')
                ),
            'repeats': (
                partial(validate_type, options=int),
                partial(validate_item, reference=1, sign='>=')
                )
            }

        # Check if no data path has been provided
        if inputs['data_path'] is None:

            # Reduce the validation map
            validation_map['data_path'] = (validation_map['data_path'][0],)

        # Check if no column information has been provided
        if inputs['columns'] is None:

            # Reduce the validation map
            validation_map['columns'] = (validation_map['columns'][0],)

        # Loop over the dictionary items
        for key, value in inputs.items():

            # Loop over the validation functions
            for function in validation_map[key]:

                # Run the validation function
                function(key, value)

            # Check if the key is 'column'
            if key == 'columns' and inputs['columns'] is not None:

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

            # Check if the key is 'holdout'
            if key == 'holdout':

                # Check if the value is a float
                if isinstance(value, float):

                    # Check if the value is outside the range
                    validate_item(key, value, 0, '>')
                    validate_item(key, value, 1, '<')

                # Else, check if the value is an integer
                elif isinstance(value, int):

                    # Check if the value is smaller than the number of splits
                    validate_item(key, value, inputs['splits'], '>=')
