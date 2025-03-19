"""Data & learning model handling."""

# Author: Tim Ortkamp

# %% External package import

from os import listdir
from os.path import isdir

# %% Internal package import

from pyanno4rt.datahub import Datahub
from pyanno4rt.learning_model.dataset import (
    EmptyDataGenerator, ImageDataGenerator, TabularDataGenerator)
from pyanno4rt.learning_model.features import FeatureCalculator
from pyanno4rt.tools import filter_dict

# %% Class definition


class DataModelHandler():
    """
    Data & learning model handling class.

    This class implements methods to handle the import of the base dataset \
    and the feature (re)calculation.

    Parameters
    ----------
    model_label : str
        Label for the machine learning model.

    model_folder_path : None or str
        Path to a folder for loading an external model.

    data_path : None or str
        Path to the dataset used for fitting the machine learning model.

    data_columns : list
        List of :class:`~pyanno4rt.learning_model.features._feature.Feature` \
        and :class:`~pyanno4rt.learning_model.features._label.Label` objects.

    tune_splits : int
        Number of splits for the stratified cross-validation within each \
        hyperparameter optimization step.

    tune_repeats : int
        Number of repeats for the stratified cross-validation within each \
        hyperparameter optimization step.

    oof_splits : int
        Number of splits for the stratified cross-validation within the \
        out-of-folds evaluation step.

    oof_repeats : int
        Number of repeats for the stratified cross-validation within the \
        out-of-folds evaluation step.

    write_features : bool
        Indicator for writing the iteratively calculated feature vectors into \
        a feature history.

    Attributes
    ----------
    model_label : str
        See 'Parameters'.

    write_features : bool
        See 'Parameters'.

    data_generator : object of class \
        :class:`~pyanno4rt.learning_model.dataset._empty_data_generator.EmptyDataGenerator`\
        :class:`~pyanno4rt.learning_model.dataset._tabular_data_generator.TabularDataGenerator`
        The object used to handle the base dataset.

    feature_calculator : object of class \
        :class:`~pyanno4rt.learning_model.features._feature_calculator.FeatureCalculator`
        The object used to (re)calculate the feature values and gradients.
    """

    def __init__(
            self,
            model_label,
            model_folder_path,
            data_path,
            data_columns,
            tune_splits,
            tune_repeats,
            oof_splits,
            oof_repeats,
            write_features):

        # Initialize the datahub
        hub = Datahub()

        # Loop over the model-related datahub attributes
        for attribute in (
                'datasets', 'feature_maps', 'model_instances',
                'model_inspections', 'model_evaluations', 'model_outcomes'):

            # Check if the attribute has not been initialized yet
            if not getattr(hub, attribute):

                # Initialize the attribute
                setattr(hub, attribute, {})

        # Get the instance attributes from the arguments
        self.model_label, self.write_features = model_label, write_features

        # Check if no data path has been passed
        if not data_path:

            # Initialize the empty dataset generator
            self.data_generator = EmptyDataGenerator(
                model_label=model_label,
                model_folder_path=model_folder_path,
                data_columns={
                    value['column']: (
                        {'type': key.lower()}
                        | filter_dict(value, remove_keys=('column',)))
                    for item in data_columns
                    for key, value in item.to_dict().items()})

        # Check if the data path leads to a tabular file
        elif data_path.endswith('.csv'):

            # Initialize the tabular dataset generator
            self.data_generator = TabularDataGenerator(
                model_label=model_label,
                data_path=data_path,
                data_columns={
                    value['column']: (
                        {'type': key.lower()}
                        | filter_dict(value, remove_keys=('column',)))
                    for item in data_columns
                    for key, value in item.to_dict().items()},
                tune_splits=tune_splits,
                tune_repeats=tune_repeats,
                oof_splits=oof_splits,
                oof_repeats=oof_repeats)

        # Check if the data path leads to a folder
        elif isdir(data_path) and all(
                any(file.endswith(extension) for extension in (
                    '.jpg', '.npy', '.npz', '.png'))
                for file in listdir(data_path)):

            # Initialize the image dataset generator
            self.data_generator = ImageDataGenerator(
                model_label=model_label,
                model_folder_path=model_folder_path)

            # Raise an error to indicate the missing implementation
            raise ValueError(
                "Image-based data generation has not been implemented yet ...")

        # Initialize the feature calculator
        self.feature_calculator = FeatureCalculator(write_features)

    def integrate(self):
        """Integrate the learning model-related classes."""

        # Generate the data information dictionary
        data_information, feature_map = self.data_generator.generate()

        # Add the static values map to the feature calculator
        self.feature_calculator.add_static_map(
            data_information['feature_statics'])

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

            # Log a message about the missing feature history
            hub.logger.display_info(
                "Feature history has not been written for "
                f"'{self.model_label}' ...")
