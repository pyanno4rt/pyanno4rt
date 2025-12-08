"""Data & learning model handling."""

# Author: Tim Ortkamp

# %% External package import

from functools import partial

# %% Internal package import

from pyanno4rt.learning.features import FeatureCalculator
from pyanno4rt.logging import get_logger
from pyanno4rt.tools import filter_dict, get_machine_learning_components
from pyanno4rt.validation import validate_type

# %% Class definition


class DataModelHandler():
    """
    Data & learning model handling class.

    This class implements methods to handle the outcome models.

    Parameters
    ----------
    handlers : list
        Dictionary with the handlers (patient, plan, dose).

    Attributes
    ----------
    handlers : dict
        See 'Parameters'.

    models : list
        Outcome models.

        Currently available:

        - :class:`~pyanno4rt.learning.models._tree._decision_tree.DecisionTree`

        - :class:`~pyanno4rt.learning.models._neighbors._k_nearest_neighbors.KNearestNeighbors`

        - :class:`~pyanno4rt.learning.models._logistic._logistic_regression.LogisticRegression`

        - :class:`~pyanno4rt.learning.models._naive_bayes._naive_bayes.NaiveBayes`

        - :class:`~pyanno4rt.learning.models._neural_network._neural_network.NeuralNetwork`

        - :class:`~pyanno4rt.learning.models._forest._random_forest.RandomForest`

        - :class:`~pyanno4rt.learning.models._svm._support_vector_machine.SupportVectorMachine`
    """

    def __init__(
            self,
            handlers):

        # Log a message about the initialization of the class
        get_logger().info("Initializing data model handler ...")

        # Validate the input arguments
        self.validate(filter_dict(vars(), remove_keys=('self',)))

        # Get the instance attributes
        self.handlers = handlers
        self.models = [
            component.model
            for component in get_machine_learning_components(
                handlers['plan_handler'].components)]

    def load_datasets(self):
        """Load the datasets for all models."""

        # Loop over the models
        for model in self.models:

            # Log a message about the dataset loading
            get_logger().info("Loading dataset for '%s' ...", model.label)

            # Load the model data
            model.load_data()

    def add_calculators(self):
        """Add the feature calculators to the models."""

        # Loop over the models
        for model in self.models:

            # Log a message about setting up the feature calculator
            get_logger().info(
                "Adding feature calculator for '%s' ...", model.label)

            # Initialize the feature calculator
            model.feature_calculator = FeatureCalculator(self.handlers)

            # Add the feature map
            model.feature_calculator.set_mapping(model.dataset.feature_map)

    def fit_models(self):
        """Fit the models."""

        # Loop over the models
        for model in self.models:

            # Log a message about the model fitting
            get_logger().info("Fitting model '%s' ...", model.label)

            # Get the features and labels
            features, labels = (
                model.dataset.feature_values, model.dataset.label_values)

            # Fit the preprocessor
            model.fit_preprocessor(features, labels)

            # Tune the hyperparameters
            model.tune_hyperparameters(features, labels)

            # Fit the model
            model.fit_predictor(features, labels)

    def validate(
            self,
            inputs):
        """
        Validate the input arguments.

        Parameters
        ----------
        inputs : dict
            Dictionary with the input arguments.
        """

        # Get the validation map
        validation_map = {
            'handlers': (
                partial(validate_type, options=dict),
                )
            }

        # Loop over the inputs
        for key, value in inputs.items():

            # Loop over the validation functions
            for function in validation_map[key]:

                # Run the validation function
                function(key, value)
