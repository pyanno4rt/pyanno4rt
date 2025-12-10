"""Anonymous dataset."""

# Author: Tim Ortkamp

# %% External package import

from functools import partial

# %% Internal package import

from pyanno4rt.learning._maps import COLUMNS, FEATURES
from pyanno4rt.logging import get_logger
from pyanno4rt.tools import filter_dict
from pyanno4rt.validation import (
    validate_length, validate_subtype, validate_type)

# %% Class definition


class AnonymousDataset():
    """
    Anonymous dataset class.

    This class provides methods to handle an anonymous base dataset (only \
    including the data columns, e.g. for a prefitted model).

    Parameters
    ----------
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
    columns : list
        See 'Parameters'.

    feature_map : dict
        Dictionary with mappings between features and calculation functions.
    """

    def __init__(
            self,
            columns):

        # Get the input arguments
        self.inputs = filter_dict(vars(), remove_keys=('self',))

        # Check the input arguments
        self.validate(self.inputs)

        # Log a message about the initialization of the class
        get_logger().info("Initializing anonymous dataset ...")

        # Get the data columns
        self.columns = columns

        # Initialize the feature map
        self.feature_map = None

    def generate(self):
        """Generate the data attributes."""

        # Build the feature map
        self._build_map()

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
            'columns': (
                partial(validate_type, options=list),
                partial(validate_subtype, options=(*COLUMNS.values(),)),
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
