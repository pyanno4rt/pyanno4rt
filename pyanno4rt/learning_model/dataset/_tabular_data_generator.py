"""Tabular dataset generation."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from functools import partial
from itertools import compress, tee
from numpy import array, logical_and, seterr, vstack, where, zeros
from math import inf
from pandas import read_csv
from sklearn.model_selection import StratifiedKFold

# %% Internal package import

from pyanno4rt.datahub import Datahub
from pyanno4rt.learning_model.features import feature_map
from pyanno4rt.tools import custom_round, deduplicate, identity, replace_nan

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

    data_path : str
        Path to the data set used for fitting the machine learning model.

    data_columns : dict
        Dictionary with the column information on features and label.

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

    data_path : str
        See 'Parameters'.

    data_columns : dict
        See 'Parameters'.

    tune_splits : int
        See 'Parameters'.

    oof_splits : int
        See 'Parameters'.
    """

    def __init__(
            self,
            model_label,
            data_path,
            data_columns,
            tune_splits,
            oof_splits):

        # Log a message about the initialization of the class
        Datahub().logger.display_info(
            f"Initializing data generator for '{model_label}' model from "
            "tabular base dataset ...")

        # Get the instance attributes from the arguments
        self.model_label = model_label
        self.data_path = data_path
        self.data_columns = data_columns
        self.tune_splits = tune_splits
        self.oof_splits = oof_splits

    def generate(self):
        """
        Generate the data information and the feature map.

        Returns
        -------
        dict
            Dictionary with the decomposed, modulated and binarized data \
            information.

        dict
            Dictionary with the mappings of feature names, segments and \
            computation/differentiation functions.
        """

        # Initialize the datahub
        hub = Datahub()

        # Decompose the base tabular dataset and the meta information
        data_information = self.decompose(
            read_csv(self.data_path), self.data_columns)

        # Check if a time-after-radiotherapy variable has been passed
        if data_information['time_variable_name']:

            # Modulate the data information
            data_information = self.modulate(data_information)

        # Binarize the data information
        data_information = self.binarize(data_information)

        # Add the fold numbers
        data_information = self.add_fold_numbers(
            data_information, self.tune_splits, self.oof_splits)

        # Enter the data information dictionary into the datahub
        hub.datasets |= {self.model_label: data_information}

        # Generate the feature map
        feature_map = self.create_map(data_information['feature_definitions'])

        # Enter the feature map into the datahub
        hub.feature_maps |= {self.model_label: feature_map}

        return data_information, feature_map

    def decompose(
            self,
            data_frame,
            data_columns):
        """
        Decompose the base tabular dataset.

        Parameters
        ----------
        data_frame : :class:`~pandas.DataFrame`
            Dataframe with the feature and label names/values.

        data_columns : dict
            Dictionary with the column information on features and label.

        Returns
        -------
        dict
            Dictionary with the decomposed data information.
        """

        # Log a message about the dataset decomposition
        Datahub().logger.display_info(
            "Decomposing tabular base dataset into features, label and time "
            "variable ...")

        # Get the meta information for the features
        feature_meta = {key: value for key, value in data_columns.items()
                        if value['type'] == 'feature'}

        # Get the meta information for the label
        label_meta = {key: value for key, value in data_columns.items()
                      if value['type'] == 'label'}

        # Get the variable names
        feature_names = list(feature_meta.keys())
        label_name = next(iter(label_meta))
        time_variable_name = label_meta[label_name].get('time_variable')

        # Get the features from the dataframe
        features = data_frame.drop(
            data_frame.columns.difference(feature_names), axis=1)

        # Return the data information dictionary
        return (
            {'raw_data': data_frame,
             'feature_names': list(features.columns),
             'feature_values': features.values,
             'feature_scales': [feature_meta[feature]['scale']
                                for feature in list(features.columns)],
             'label_name': label_name,
             'label_values': data_frame[label_name].values,
             'label_bounds': label_meta[label_name].get('bounds', [1, 1]),
             'label_viewpoint': label_meta[label_name].get(
                 'viewpoint', 'longitudinal'),
             'time_variable_name': time_variable_name,
             'time_variable_values': (
                 data_frame[filter(None, [time_variable_name])].values)}
            | {'feature_statics': {
                key: feature_meta[key]['value']
                for key in list(features.columns)
                if feature_meta[key].get('value')}}
            | {'feature_definitions': {
                key: {'segment': feature_meta[key].get('segment'),
                      'function': feature_meta[key].get('function'),
                      'argument': feature_meta[key].get('argument')}
                for key in list(features.columns)}})

    def modulate(
            self,
            data_information):
        """
        Modulate the data information.

        Parameters
        ----------
        data_information : dict
            Dictionary with the decomposed data information.

        Returns
        -------
        dict
            Dictionary with the modulated data information.
        """

        # Log a message about the dataset modulation
        Datahub().logger.display_info(
            "Modulating data information by feature "
            f"'{data_information['time_variable_name']}' for label viewpoint "
            f"'{data_information['label_viewpoint']}' ...")

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
        if data_information['label_viewpoint'] != 'longitudinal':

            # Get the mapping between patient features and sample indices
            patient_map = deduplicate(
                map(tuple, data_information['feature_values']))

            # Overwrite the feature values by the patient features
            data_information['feature_values'] = array((*patient_map,))

            # Overwrite the label values by the squeezed labels
            data_information['label_values'] = vstack(tuple(map(
                partial(squeeze_labels, index_sets=patient_map.values()),
                zip(*viewpoints[data_information['label_viewpoint']])))).T

            # Check if the label values are single
            if data_information['label_values'].shape[1] == 1:

                # Reshape the label values into 1D
                data_information['label_values'] = (
                    data_information['label_values'].reshape(-1))

        # Check if the label viewpoint is 'profile'
        if data_information['label_viewpoint'] == 'profile':

            # Overwrite the label name by a list of generic strings
            data_information['label_name'] = [
                f"{data_information['label_name']}_{i}"
                for i in range(data_information['label_values'].shape[1])]

        return data_information

    def binarize(
            self,
            data_information):
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

        # Transform the label bounds by replacing None with limit values
        label_bounds = [
            data_information['label_bounds'][index]
            if data_information['label_bounds'][index] is not None
            else (-1)**(index+1)*inf for index in range(2)]

        # Log a message about the dataset binarization
        Datahub().logger.display_info(
            f"Binarizing data information by label bounds {label_bounds} ...")

        # Get the label values
        label_values = data_information['label_values']

        # Overwrite the label values with the binarizations
        data_information['label_values'] = where(
            (label_values >= label_bounds[0])
            & (label_values <= label_bounds[1]), 1, 0)

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

    def create_map(
            self,
            definitions):
        """
        Create the feature map.

        Parameters
        ----------
        definitions : dict
            Dictionary with the mappings of feature names, segments and \
            string functions.

        Returns
        -------
        dict
            Dictionary with the mappings of feature names, segments and \
            computation/differentiation functions.
        """

        def get_single_definition(key):
            """Get the mapping for a single definition."""

            # Get the feature definition as string
            definition = feature_map[definitions[key]['function']]

            # Get the argument of the feature definition
            args = definitions[key].get('argument')

            # Return the single feature map
            return {key: {
                'segment': definitions[key]['segment'],
                'class': definition.feature_class,
                'computation': methods[args is None](definition.compute, args),
                'differentiation': (
                    methods[args is None](definition.differentiate, args)
                    if definition.feature_class == 'Dosiomics' else None)}}

        # Create a boolean mapping to the internal functions
        methods = {True: identity, False: partial}

        return {key: value
                for item in map(get_single_definition, definitions.keys())
                for key, value in item.items()}
