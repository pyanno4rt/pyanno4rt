"""Data & learning model handling."""

# %% External package import

from math import inf

# %% Internal package import

from pyanno4rt.datahub import Datahub
from pyanno4rt.learning_model.data import Dataset
from pyanno4rt.learning_model.features import (
    FeatureMapGenerator, FeatureCalculator)

# %% Class definition


class DataModelHandler():
    """
    Data & learning model handling class.

    This class implements methods to handle the base data set, generate the \
    feature map (associating each feature with the corresponding segment and \
    definition from the feature catalogue) and initialize the feature \
    calculator for the parent learning model.

    Parameters
    ----------
    data_path : string
        Path to the data set used for fitting the learning models.

    feature_filter : tuple or list
        A (sub)set of the feature names as an iterable and a value from \
        {'retain', 'remove'} as an indicator for retaining or removing the \
        (sub)set prior to the modeling process, all features are retained if \
        no value is passed.

    label_viewpoint : {'early', 'late', 'long-term', 'longitudinal', 'profile'}
        Time of observation for the presence of tumor control and/or normal \
        tissue complication events. The values can be described as follows:

        - 'early' : observation period between 0 and 6 months after treatment;
        - 'late' : observation period between 6 and 15 months after treatment;
        - 'long-term' : observation period between 15 and 24 months after \
           treatment;
        - 'longitudinal' : no observation period, time after treatment is \
           included as a covariate;
        - 'profile' : TCP/NTCP profiling over time, multi-label scenario \
           with one label per month (up to 24 labels in total).

    label_bounds : tuple or list
        Bounds for the label values to binarize them into positive and \
        negative class, i.e., label values within the specified bounds will \
        be interpreted as a binary 1.

    fuzzy_matching : bool
        Indicator for the use of fuzzy string matching (if False, exact \
        string matching is applied) to generate the mapping between features, \
        segmented structures and calculation functions. Only used when \
        data-driven components are present.

    write_features : bool
        Indicator for writing the iteratively computed feature vectors to the \
        feature history. Only used when data-driven components are present.

    write_gradients : bool
        Indicator for writing the iteratively computed feature gradient \
        matrices to the gradient history. Only used when data-driven \
        components are present.

    Attributes
    ----------
    data : object of class `Dataset`
        Instance of the class `Dataset`, which provides a preprocessed \
        version of the raw dataset along with its individual components.

    feature_map_generator : object of class `FeatureMapGenerator`
        Instance of the class `FeatureMapGenerator`, which generates a \
        feature map that links each feature to its related segment and \
        computation/differentiation function. This is based on fuzzy or exact \
        string matching, depending on the boolean value of the parameter \
        ``fuzzy_string_matching``.

    feature_calculator : object of class `FeatureCalculator`
        Instance of the class `FeatureCalculator`, which holds methods to \
        (re-)compute the feature vector and gradient matrix for changing dose \
        input. It may also store histories of features and gradients, \
        depending on the values of the parameters ``write_feat`` and \
        ``write_grad``.
    """

    def __init__(
            self,
            model_label,
            data_path,
            feature_filter,
            label_viewpoint,
            label_bounds,
            fuzzy_matching,
            write_features,
            write_gradients):

        # Initialize the datahub
        hub = Datahub()

        # Get the model label from the argument
        self.model_label = model_label

        # Check if the lower label bound is None
        if label_bounds[0] is None:

            # Replace the lower bound default with negative infinity
            label_bounds[0] = -inf

        # Check if the upper label bound is None
        if label_bounds[1] is None:

            # Replace the upper bound default with infinity
            label_bounds[1] = inf

        # Initialize the dataset
        self.data = Dataset(
            model_label=model_label, data_path=data_path,
            feature_filter=feature_filter, label_viewpoint=label_viewpoint,
            label_bounds=label_bounds)

        # Generate the feature map
        self.feature_map_generator = FeatureMapGenerator(
            model_label, hub.datasets[model_label], fuzzy_matching)

        # Initialize the feature calculator
        self.feature_calculator = FeatureCalculator(
            hub.feature_maps[model_label], write_features,
            write_gradients)

    def process_histories(
            self,
            label):
        """
        Process the feature and gradient histories from the feature \
        calculator.

        Parameters
        ----------
        label : string
            Label for the model.
        """
        # Initialize the datahub
        hub = Datahub()

        # Check if the feature history is non-existent
        if not hasattr(self.feature_calculator, 'feature_history'):

            # Log a message about the unavailability of the feature history
            hub.logger.display_info("Feature history retrieval is not "
                                    "performed for '{}' ..."
                                    .format(label))
        else:
            # Convert the feature history array into a dictionary
            self.feature_calculator.feature_history = dict(
                zip((*hub.feature_maps[self.model_label],),
                    (*self.feature_calculator.feature_history[1:, :]
                     .transpose(),)))

        # Check if the gradient history is non-existent
        if not hasattr(self.feature_calculator, 'gradient_history'):

            # Log a message about the unavailability of the gradient history
            hub.logger.display_info("Gradient history retrieval is not "
                                    "performed for '{}' ..."
                                    .format(label))

            # Note: changing the gradient history format is not necessary
