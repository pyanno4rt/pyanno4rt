"""Tabular dataset generation."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from functools import partial
from itertools import compress, tee
from numpy import array, logical_and, seterr, vstack, where, zeros
from pandas import read_csv
from sklearn.model_selection import StratifiedKFold

# %% Internal package import

from pyanno4rt.datahub import Datahub
from pyanno4rt.tools import custom_round, deduplicate, replace_nan

# %% Set package options

seterr(divide='ignore', invalid='ignore')

# %% Class definition


class TabularDataGenerator():
    """
    Tabular dataset generation class.

    This class provides methods to load, decompose, modulate and binarize a \
    tabular base dataset.

    Parameters
    ----------
    model_label : str
        Label for the machine learning model.

    feature_filter : dict
        Dictionary with a list of feature names and a value from \
        {'retain', 'remove'} as an indicator for retaining/removing the \
        features prior to model fitting.

    label_name : str
        Name of the label variable.

    label_bounds : list
        Bounds for the label values to binarize into positive (value lies \
        inside the bounds) and negative class (value lies outside the \
        bounds).

    time_variable_name : str
        Name of the time-after-radiotherapy variable (unit should be days).

    label_viewpoint : {'early', 'late', 'long-term', 'longitudinal', 'profile'}
        Time of observation for the presence of tumor control and/or \
        normal tissue complication events.

    tune_splits : int
        Number of splits for the stratified cross-validation within each \
        model hyperparameter optimization step.

    oof_splits : int
        Number of splits for the stratified cross-validation within the \
        out-of-folds model evaluation step.

    Attributes
    ----------
    model_label : str
        See 'Parameters'.

    feature_filter : dict
        See 'Parameters'.

    label_name : str
        See 'Parameters'.

    label_bounds : list
        See 'Parameters'.

    time_variable_name : str
        See 'Parameters'.

    label_viewpoint : {'early', 'late', 'long-term', 'longitudinal', 'profile'}
        See 'Parameters'.

    tune_splits : int
        See 'Parameters'.

    oof_splits : int
        See 'Parameters'.
    """

    def __init__(
            self,
            model_label,
            feature_filter,
            label_name,
            label_bounds,
            time_variable_name,
            label_viewpoint,
            tune_splits,
            oof_splits):

        # Log a message about the initialization of the class
        Datahub().logger.display_info(
            f"Initializing data generator for '{model_label}' model from "
            "tabular base dataset ...")

        # Get the instance attributes from the arguments
        self.model_label = model_label
        self.feature_filter = feature_filter
        self.label_name = label_name
        self.label_bounds = label_bounds
        self.time_variable_name = time_variable_name
        self.label_viewpoint = label_viewpoint
        self.tune_splits = tune_splits
        self.oof_splits = oof_splits

    def generate(
            self,
            data_path):
        """
        Generate the data information.

        Parameters
        ----------
        data_path : str
            Path to the data set used for fitting the machine learning model.

        Returns
        -------
        dict
            Dictionary with the decomposed, modulated and binarized data \
            information.
        """

        # Decompose the base tabular dataset
        data_information = self.decompose(
            read_csv(data_path), self.feature_filter, self.label_name,
            self.time_variable_name)

        # Check if a time-after-radiotherapy variable has been passed
        if self.time_variable_name:

            # Modulate the data information
            data_information = self.modulate(
                data_information, self.label_viewpoint)

        # Binarize the data information
        data_information = self.binarize(data_information, self.label_bounds)

        # Add the fold numbers
        data_information = self.add_fold_numbers(
            data_information, self.tune_splits, self.oof_splits)

        # Enter the data information dictionary into the datahub
        Datahub().datasets |= {self.model_label: data_information}

        return data_information

    def decompose(
            self,
            dataset,
            feature_filter,
            label_name,
            time_variable_name):
        """
        Decompose the base tabular dataset.

        Parameters
        ----------
        dataset : :class:`~pandas.DataFrame`
            Dataframe with the feature and label names/values.

        feature_filter : dict
            Dictionary with a list of feature names and a value from \
            {'retain', 'remove'} as an indicator for retaining/removing the \
            features prior to model fitting.

        label_name : str
            Name of the label variable.

        time_variable_name : str
            Name of the time-after-radiotherapy variable (unit should be days).

        Returns
        -------
        dict
            Dictionary with the decomposed data information.
        """

        # Log a message about the dataset decomposition
        Datahub().logger.display_info(
            "Decomposing tabular base dataset into features, label and time "
            "variable ...")

        # Get the features from the dataset
        features = dataset.drop(
            filter(None, [label_name, time_variable_name]), axis=1)

        # Check if the filter mode is set to 'retain'
        if feature_filter['filter_mode'] == 'retain':

            # Retain the features from the filter list
            features.drop(
                features.columns.difference(feature_filter['features']),
                axis=1, inplace=True)

        else:

            # Remove the features from the filter list
            features.drop(feature_filter['features'], axis=1, inplace=True)

        # Define the output dictionary keys
        keys = ('raw_data', 'feature_names', 'feature_values', 'label_name',
                'label_values', 'time_variable_name', 'time_variable_values')

        # Define the output dictionary values
        values = (dataset, list(features.columns), features.values, label_name,
                  dataset[label_name].values, time_variable_name,
                  dataset[filter(None, [time_variable_name])].values)

        return dict(zip(keys, values))

    def modulate(
            self,
            data_information,
            label_viewpoint):
        """
        Modulate the data information.

        Parameters
        ----------
        data_information : dict
            Dictionary with the decomposed data information.

        label_viewpoint : {'early', 'late', 'long-term', 'longitudinal', \
                           'profile'}
            Time of observation for the presence of tumor control and/or \
            normal tissue complication events.

        Returns
        -------
        dict
            Dictionary with the modulated data information.
        """

        # Log a message about the dataset modulation
        Datahub().logger.display_info(
            "Modulating data information by feature "
            f"'{data_information['time_variable_name']}' for label viewpoint "
            f"'{label_viewpoint}' ...")

        def squeeze_labels(bounds, index_sets):
            """Squeeze the labels per patient by the time bounds."""

            # Get the time variable values
            times = data_information['time_variable_values']

            # Get the label values
            labels = data_information['label_values']

            # Get a boolean mask indicating interior samples per patient
            interior_mask = tee(
                (logical_and(bounds[0]*365/12 <= times[index_set],
                             bounds[1]*365/12 > times[index_set]).reshape(-1)
                 for index_set in index_sets), 2)

            # Get the label values from the interior samples per patient
            interior_labels = (labels[list(compress(value[0], value[1]))]
                               for value in zip(index_sets, interior_mask[0]))

            # Get the mean interior label value per patient
            interior_means = replace_nan(
                (numerator/denominator for numerator, denominator in zip(
                    map(sum, interior_labels), map(sum, interior_mask[1]))),
                0.0)

            return array(list(map(custom_round, interior_means)))

        # Map the label viewpoints to the time bounds
        viewpoints = {'early': ((0,), (6,)),
                      'late': ((6,), (15,)),
                      'long-term': ((15,), (24,)),
                      'longitudinal': ((), ()),
                      'profile': (range(24), range(1, 25))}

        # Check if the label viewpoint is 'longitudinal'
        if label_viewpoint != 'longitudinal':

            # Get the mapping between patient features and sample indices
            patient_map = deduplicate(
                map(tuple, data_information['feature_values']))

            # Overwrite the feature values by the patient features
            data_information['feature_values'] = array((*patient_map,))

            # Overwrite the label values by the squeezed labels
            data_information['label_values'] = vstack(tuple(map(
                partial(squeeze_labels, index_sets=patient_map.values()),
                zip(*viewpoints[label_viewpoint])))).T

            # Check if the label values are single
            if data_information['label_values'].shape[1] == 1:

                # Reshape the label values into 1D
                data_information['label_values'] = (
                    data_information['label_values'].reshape(-1))

        # Check if the label viewpoint is 'profile'
        if label_viewpoint == 'profile':

            # Overwrite the label name by a list of generic strings
            data_information['label_name'] = [
                f"{data_information['label_name']}_{i}"
                for i in range(data_information['label_values'].shape[1])]

        # Add the label viewpoint to the data information
        data_information |= {'label_viewpoint': label_viewpoint}

        return data_information

    def binarize(
            self,
            data_information,
            label_bounds):
        """
        Binarize the data information.

        Parameters
        ----------
        data_information : dict
            Dictionary with the decomposed data information.

        label_bounds : list
            Bounds for the label values to binarize into positive (value lies \
            inside the bounds) and negative class (value lies outside the \
            bounds).

        Returns
        -------
        dict
            Dictionary with the binarized data information.
        """

        # Log a message about the dataset binarization
        Datahub().logger.display_info(
            f"Binarizing data information by label bounds {label_bounds} ...")

        # Get the label values
        label_values = data_information['label_values']

        # Overwrite the label values with the binarizations
        data_information['label_values'] = where(
            (label_values >= label_bounds[0])
            & (label_values <= label_bounds[1]), 1, 0)

        # Add the label bounds to the data information
        data_information |= {'label_bounds': label_bounds}

        return data_information

    def add_fold_numbers(
            self,
            data_information,
            tune_splits,
            oof_splits):
        """
        Add the stratified cross-validation fold numbers.

        Parameters
        ----------
        data_information : dict
            Dictionary with the preprocessed data information.

        tune_splits : int
            Number of splits for the stratified cross-validation within each \
            model hyperparameter optimization step.

        oof_splits : int
            Number of splits for the stratified cross-validation within the \
            out-of-folds model evaluation step.

        Returns
        -------
        dict
            Dictionary with the stratified cross-validation fold numbers.
        """

        # Log a message about the fold number addition
        Datahub().logger.display_info(
            "Adding fold numbers for stratified cross validation ...")

        def get_folds(number_of_splits):
            """Get the fold numbers for a number of splits."""

            # Clamp the number of splits
            number_of_splits = min(
                number_of_splits, sum(data_information['label_values']))

            # Initialize the stratified k-fold cross-validator
            cross_validator = StratifiedKFold(
                n_splits=number_of_splits, random_state=4, shuffle=True)

            # Initialize the fold numbers
            folds = zeros(data_information['label_values'].shape)

            # Loop over the cross-validation splits
            for number, (_, validation_index) in enumerate(
                    cross_validator.split(data_information['feature_values'],
                                          data_information['label_values'])):

                # Enter the fold number for the current validation set
                folds[validation_index] = int(number)

            return folds

        # Add the fold numbers to the data information
        data_information |= {'tune_folds': get_folds(tune_splits),
                             'oof_folds': get_folds(oof_splits)}

        return data_information
