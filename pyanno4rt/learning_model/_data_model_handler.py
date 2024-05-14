"""Data & learning model handling."""

# %% External package import

from math import inf

# %% Internal package import

from pyanno4rt.datahub import Datahub
from pyanno4rt.learning_model.dataset import TabularDataGenerator
from pyanno4rt.learning_model.features import (
    FeatureMapGenerator, FeatureCalculator)

# %% Class definition


class DataModelHandler():
    """
    Data & learning model handling class.

    This class implements methods to handle the integration of the base \
    dataset, the feature map generator and the feature calculator.

    Parameters
    ----------
    data_path : str
        Path to the data set used for fitting the machine learning model.

    feature_filter : dict
        Dictionary with a list of feature names and a value from \
        {'retain', 'remove'} as an indicator for retaining/removing the \
        features prior to model fitting.

    label_bounds : list
        Bounds for the label values to binarize into positive (value lies \
        inside the bounds) and negative class (value lies outside the bounds).

    label_viewpoint : {'early', 'late', 'long-term', 'longitudinal', \
                       'profile'}
        Time of observation for the presence of tumor control and/or normal \
        tissue complication events.

    tune_splits : int
        Number of splits for the stratified cross-validation within each \
        hyperparameter optimization step.

    oof_splits : int
        Number of splits for the stratified cross-validation within the \
        out-of-folds evaluation step.

    fuzzy_matching : bool
        Indicator for the use of fuzzy string matching to generate the \
        feature map (if False, exact string matching is applied).

    write_features : bool
        Indicator for writing the iteratively calculated feature vectors into \
        a feature history.

    Attributes
    ----------
    model_label : str
        See 'Parameters'.

    data_path : str
        See 'Parameters'.

    write_features : bool
        See 'Parameters'.

    dataset : object of class \
        :class:`~pyanno4rt.learning_model.dataset._tabular_data_generator.TabularDataGenerator`
        The object used to handle the base dataset.

    feature_map_generator : object of class \
        :class:`~pyanno4rt.learning_model.features._feature_map_generator.FeatureMapGenerator`
        The object used to map the dataset features to the feature definitions.

    feature_calculator : object of class \
        :class:`~pyanno4rt.learning_model.features._feature_calculator.FeatureCalculator`
        The object used to (re-)calculate the feature values and gradients.
    """

    def __init__(
            self,
            model_label,
            data_path,
            feature_filter,
            label_name,
            label_bounds,
            time_variable_name,
            label_viewpoint,
            tune_splits,
            oof_splits,
            fuzzy_matching,
            write_features):

        # Initialize the datahub
        hub = Datahub()

        # Loop over the model-related datahub attributes
        for attribute in ('datasets', 'feature_maps', 'model_instances',
                          'model_inspections', 'model_evaluations'):

            # Check if the attribute has not been initialized yet
            if not getattr(hub, attribute):

                # Initialize the attribute
                setattr(hub, attribute, {})

        # Get the instance attributes from the arguments
        self.model_label = model_label
        self.data_path = data_path
        self.write_features = write_features

        # Check if the data path leads to a tabular data file
        if data_path.endswith('.csv'):

            # Transform the label bounds by replacing None with limit values
            label_bounds = [
                label_bounds[index] if label_bounds[index] is not None
                else (-1)**(index+1)*inf for index in range(2)]

            # Initialize the tabular dataset generator
            self.dataset = TabularDataGenerator(
                model_label=model_label,
                feature_filter=feature_filter,
                label_name=label_name,
                label_bounds=label_bounds,
                time_variable_name=time_variable_name,
                label_viewpoint=label_viewpoint,
                tune_splits=tune_splits,
                oof_splits=oof_splits)

        # Initialize the feature map generator
        self.feature_map_generator = FeatureMapGenerator(
            model_label, fuzzy_matching)

        # Initialize the feature calculator
        self.feature_calculator = FeatureCalculator(write_features)

    def integrate(self):
        """Integrate the learning model-related classes."""

        # Generate the data information dictionary
        data_information = self.dataset.generate(self.data_path)

        # Generate the feature map
        feature_map = self.feature_map_generator.generate(data_information)

        # Add the feature map to the feature calculator
        self.feature_calculator.add_feature_map(feature_map)

    def process_feature_history(self):
        """Process the feature history from the feature calculator."""

        # Initialize the datahub
        hub = Datahub()

        # Check if the feature history has been written
        if self.write_features:

            # Transform the feature history into a dictionary
            self.feature_calculator.feature_history = dict(zip(
                (*hub.feature_maps[self.model_label],),
                (*self.feature_calculator.feature_history[1:, :].transpose(),)
                ))

        else:

            # Log a message about the non-writing of the feature history
            hub.logger.display_info("Feature history has not been written for "
                                    f"'{self.model_label}' ...")
