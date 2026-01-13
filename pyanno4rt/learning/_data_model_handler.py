"""Data model handling."""

# Author: Tim Ortkamp

# %% External package import

from functools import partial

# %% Internal package import

from pyanno4rt.learning.features import FeatureCalculator
from pyanno4rt.logging import get_logger
from pyanno4rt.tools import filter_dict, get_machine_learning_components
from pyanno4rt.validation import validate_length, validate_type

# %% Class definition


class DataModelHandler():
    """
    Data model handling class.

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

        - :class:`~pyanno4rt.learning.models._forest._random_forest.RandomForest`

        - :class:`~pyanno4rt.learning.models._logistic._logistic_regression.LogisticRegression`

        - :class:`~pyanno4rt.learning.models._naive_bayes._naive_bayes.NaiveBayes`

        - :class:`~pyanno4rt.learning.models._neighbors._k_nearest_neighbors.KNearestNeighbors`

        - :class:`~pyanno4rt.learning.models._neural_network._feed_forward_net.FeedForwardNet`

        - :class:`~pyanno4rt.learning.models._svm._support_vector_machine.SupportVectorMachine`

        - :class:`~pyanno4rt.learning.models._tree._decision_tree.DecisionTree`

    outcomes : dict
        Dictionary with the outcome results.
    """

    def __init__(
            self,
            handlers):

        # Validate the input arguments
        self.validate(filter_dict(vars(), remove_keys=('self',)))

        # Log a message about the initialization of the class
        get_logger().info("Initializing data model handler ...")

        # Get the instance attributes
        self.handlers = handlers
        self.models = [
            component.model for component in get_machine_learning_components(
                handlers['plan_handler'].components)]

        # Initialize the outcome dictionary
        self.outcomes = {}

        # Check if no models have been provided
        if len(self.models) == 0:

            # Log a message about the missing models
            get_logger().warning(
                "Treatment plan does not include model-based components - "
                "outcome modeling is skipped ...")

    def load_datasets(self):
        """Load the datasets for the models."""

        # Loop over the models
        for model in (model for model in self.models if model.reload_data):

            # Load the model data
            model.load_data()

            # Update the refreshing indicator
            model.reload_data = False

    def set_calculators(self):
        """Set the feature calculators for the models."""

        # Loop over the models
        for model in (model for model in self.models if model.reset_calc):

            # Add the feature calculator
            model.add_calculator(FeatureCalculator(self.handlers))

            # Update the refreshing indicator
            model.reset_calc = False

    def fit_models(self):
        """Fit the models."""

        # Loop over the models
        for model in (model for model in self.models if model.refit):

            # Check if no model path has been provided
            if model.model_path is None:

                # Log a message about fitting the model
                get_logger().info("Fitting model '%s' ...", model.label)

                # Get the features and labels
                features, labels = (
                    model.dataset.feature_values, model.dataset.label_values)

                # Check if a preprocessor has been provided
                if model.preprocessor is not None:

                    # Log a message about the preprocessing pipeline
                    get_logger().info(
                        "Building preprocessing pipeline 'Input -> %s -> "
                        "Output' for model '%s' ...",
                        ' → '.join(
                            step.name for step in model.preprocessor.pipeline),
                        model.label)

                # Fit the preprocessor
                model.fit_preprocessor(features, labels)

                # Tune the hyperparameters
                model.tune_hyperparameters(features, labels)

                # Fit the model
                model.fit_predictor(
                    *model.preprocess(features, labels, mode='fit'))

            else:

                # Load the model
                model.load()

            # Update the refreshing indicator
            model.refit = False

        # Loop over the machine learning components
        for component in get_machine_learning_components(
                self.handlers['plan_handler'].components):

            # Update the model parameters
            component.update_from_model()

    def inspect_models(self):
        """Inspect the models."""

        # Loop over the models
        for model in (model for model in self.models if model.reinspect):

            # Inspect the model
            model.inspect()

            # Update the refreshing indicator
            model.reinspect = False

    def evaluate_models(self):
        """Evaluate the models."""

        # Loop over the models
        for model in (model for model in self.models if model.reevaluate):

            # Evaluate the model
            model.evaluate()

            # Update the refreshing indicator
            model.reevaluate = False

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
                partial(validate_length, reference=3, sign='==')
                )
            }

        # Loop over the inputs
        for key, value in inputs.items():

            # Loop over the validation functions
            for function in validation_map[key]:

                # Run the validation function
                function(key, value)
