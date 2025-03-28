"""Model parameters."""

# Author: Tim Ortkamp

# %% External package import

from copy import deepcopy
from functools import partial
from os.path import abspath

# %% Internal package import

from pyanno4rt.checking import (
    check_key_in_dict, check_length, check_path, check_regular_extension,
    check_regular_extension_directory, check_subtype, check_type,
    check_value, check_value_in_set)
import pyanno4rt.learning._maps as maps
from pyanno4rt.tools import filter_dict

# %% Class definitions


class ModelParameters():
    """
    Learning model parameters class.

    This class provides methods to set, validate and serialize the parameter \
    set for the learning models.

    Parameters
    ----------
    model_label : str
        Label for the learning model.

    model_folder_path : None or str
        Path to an external model folder.

    data_path : None or str
        Path to the data set used for fitting the learning model.

    data_columns : list, default=[]
        List of \
        :class:`~pyanno4rt.learning.features._columns.DynamicFeature` \
        or :class:`~pyanno4rt.learning.features._columns.StaticFeature` \
        and :class:`~pyanno4rt.learning.features._columns.Label` objects.

    preprocessing : list, default=['Identity']
        Sequence of labels associated with data preprocessing steps.

        Currently available:

        - 'Identity' \
            :class:`~pyanno4rt.learning.preprocessing._identity.Identity`
        - 'StandardScaler' \
            :class:`~pyanno4rt.learning.preprocessing._standard_scaler.StandardScaler`
        - 'Whitening' \
            :class:`~pyanno4rt.learning.preprocessing._whitening.Whitening`

    architecture : {'vanilla-input-convex', 'vanilla'}, default='vanilla'
        Type of architecture for a neural network model.

    max_hidden_layers : int, default=2
        Maximum number of hidden layers for a neural network model.

    tune_space : dict, default={}
        Search space for the Bayesian hyperparameter optimization.

    tune_evaluations : int, default=50
        Number of evaluation steps (trials) for the Bayesian \
        hyperparameter optimization.

    tune_score : {'AUC', 'Brier score', 'Logloss'}, default='Logloss'
        Scoring function for the evaluation of the hyperparameter set \
        candidates.

    tune_splits : int, default=5
        Number of splits for the data resampling within each hyperparameter \
        optimization step.

    tune_repeats : int, default=1
        Number of repeats for the data resampling within each hyperparameter \
        optimization step.

    inspect : bool, default=False
        Indicator for the inspection of the learning model.

    evaluate : bool, default=False
        Indicator for the evaluation of the learning model.

    oof_splits : int, default=5
        Number of splits for the data resampling within the out-of-folds \
        evaluation step.

    oof_repeats : int, default=1
        Number of repeats for the data resampling within the out-of-folds \
        evaluation step.

    write_features : bool, default=False
        Indicator for writing a history of the iteration-wise feature vectors.

    display_options : dict, \
        default={'graphs': ['AUC-ROC', 'AUC-PR', 'F1'],\
                 'kpis': ['Logloss', 'Brier score', 'Subset accuracy', \
                          'Cohen Kappa', 'Hamming loss', 'Jaccard score', \
                          'Precision', 'Recall', 'F1 score', 'MCC', 'AUC']}
        Dictionary with the graph and KPI display options.

    Attributes
    ----------
    model_label : str
        See 'Parameters'.

    model_folder_path : None or str
        See 'Parameters'.

    data_path : None or str
        See 'Parameters'.

    data_columns : list
        See 'Parameters'.

    preprocessing : list
        See 'Parameters'.

    architecture : {'vanilla-input-convex', 'vanilla'}
        See 'Parameters'.

    max_hidden_layers : int
        See 'Parameters'.

    tune_space : dict
        See 'Parameters'.

    tune_evaluations : int
        See 'Parameters'.

    tune_score : {'AUC', 'Brier score', 'Logloss'}
        See 'Parameters'.

    tune_splits : int
        See 'Parameters'.

    tune_repeats : int
        See 'Parameters'.

    inspect : bool
        See 'Parameters'.

    evaluate : bool
        See 'Parameters'.

    oof_splits : int
        See 'Parameters'.

    oof_repeats : int
        See 'Parameters'.

    write_features : bool
        See 'Parameters'.

    display_options : dict
        See 'Parameters'.
    """

    def __init__(
            self,
            model_label,
            model_folder_path=None,
            data_path=None,
            data_columns=[],
            preprocessing=['Identity'],
            architecture='vanilla',
            max_hidden_layers=2,
            tune_space={},
            tune_evaluations=50,
            tune_score='Logloss',
            tune_splits=5,
            tune_repeats=1,
            inspect=False,
            evaluate=False,
            oof_splits=5,
            oof_repeats=1,
            write_features=False,
            display_options={
                'graphs': ['AUC-ROC', 'AUC-PR', 'F1'],
                'kpis': [
                    'Logloss', 'Brier score', 'Subset accuracy',
                    'Cohen Kappa', 'Hamming loss', 'Jaccard score',
                    'Precision', 'Recall', 'F1 score', 'MCC', 'AUC']}):

        # Get the input arguments
        self.inputs = filter_dict(vars(), remove_keys=('self',))

        # Check the input arguments
        self.check(self.inputs | tune_space | display_options)

        # Loop over the input arguments
        for item in self.inputs.items():

            # Set the attribute
            setattr(self, *item)

        # Check if a model folder path has been passed
        if self.model_folder_path is not None:

            # Convert the model folder path into an absolute path
            self.model_folder_path = abspath(self.model_folder_path)

        # Check if a data path has been passed
        if self.data_path is not None:

            # Convert the data path into an absolute path
            self.data_path = abspath(self.data_path)

    def to_dict(self):
        """Serialize the model parameters into a dictionary."""

        # Get the parameter dictionary
        dictionary = deepcopy(self.inputs)

        # Serialize the data columns
        dictionary['data_columns'] = [
            item.to_dict() for item in dictionary['data_columns']]

        return dictionary

    @classmethod
    def from_dict(
            cls,
            dictionary):
        """
        Deserialize the model parameters from a dictionary.

        Parameters
        ----------
        dictionary : dict
            Dictionary with the model parameters.

        Returns
        -------
        object of class \
            :class:`~pyanno4rt.learning._model_parameters.ModelParameters`
            The object used to handle the model parameters.
        """

        # Deserialize the data columns
        dictionary['data_columns'] = [
            maps.COLUMNS[key].from_dict(value)
            for item in dictionary['data_columns']
            for key, value in item.items()]

        return cls(**dictionary)

    def check(
            self,
            inputs):
        """
        Check the input arguments.

        Parameters
        ----------
        inputs : dict
            Dictionary with the mappings between argument names and values.
        """

        check_map = {
            'model_label': (
                partial(check_type, types=str),),
            'model_folder_path': (
                partial(check_type, types=(type(None), str)),
                partial(check_path)),
            'data_path': (
                partial(check_type, types={
                    True: (type(None), str), False: str},
                    type_condition=isinstance(
                        inputs.get('model_folder_path'), str)),
                partial(check_regular_extension, extensions=('.csv',)),
                partial(check_regular_extension_directory, extensions=(
                    '.jpg', '.npy', '.npz', '.png'), no_directory=('.csv',))),
            'data_columns': (
                partial(check_type, types={
                    True: (type(None), list), False: list},
                    type_condition=isinstance(
                        inputs.get('model_folder_path'), str)),
                partial(check_length, reference=2, sign='>=')),
            'preprocessing': (
                partial(check_type, types=list),
                partial(check_subtype, types=str),
                partial(check_value_in_set, options=tuple(maps.TRANSFORMERS))),
            'architecture': (
                partial(check_type, types=str),
                partial(check_value_in_set, options=(
                    'vanilla', 'vanilla input-convex'))),
            'max_hidden_layers': (
                partial(check_type, types=int),
                partial(check_value, reference=0, sign='>=')),
            'tune_space': (
                partial(check_type, types=dict),),
            'tune_evaluations': (
                partial(check_type, types=int),
                partial(check_value, reference=0, sign='>')),
            'tune_score': (
                partial(check_type, types=str),
                partial(check_value_in_set, options=tuple(
                    ('AUC', *maps.LOSSES)))),
            'tune_splits': (
                partial(check_type, types=int),
                partial(check_value, reference=1, sign='>=')),
            'tune_repeats': (
                partial(check_type, types=int),
                partial(check_value, reference=1, sign='>=')),
            'inspect': (
                partial(check_type, types=bool),),
            'evaluate': (
                partial(check_type, types=bool),),
            'oof_splits': (
                partial(check_type, types=int),
                partial(check_value, reference=1, sign='>=')),
            'oof_repeats': (
                partial(check_type, types=int),
                partial(check_value, reference=1, sign='>=')),
            'write_features': (
                partial(check_type, types=bool),),
            'display_options': (
                partial(check_type, types=dict),
                partial(check_key_in_dict, keys=('graphs', 'kpis'))),
            'criterion': (
                partial(check_type, types=list),
                partial(check_value_in_set, options=('entropy', 'gini'))),
            'splitter': (
                partial(check_type, types=list),
                partial(check_value_in_set, options=('best', 'random'))),
            'max_depth': (
                partial(check_type, types=list),
                partial(check_subtype, types=int),
                partial(check_value, reference=0, sign='>', is_vector=True)),
            'min_samples_split': (
                partial(check_type, types=list),
                partial(check_length, reference=2, sign='=='),
                partial(check_subtype, types=float),
                partial(check_value, reference=0, sign='>=', is_vector=True),
                partial(check_value, reference=1, sign='<=', is_vector=True)),
            'min_samples_leaf': (
                partial(check_type, types=list),
                partial(check_length, reference=2, sign='=='),
                partial(check_subtype, types=float),
                partial(check_value, reference=0, sign='>=', is_vector=True),
                partial(check_value, reference=1, sign='<=', is_vector=True)),
            'min_weight_fraction_leaf': (
                partial(check_type, types=list),
                partial(check_length, reference=2, sign='=='),
                partial(check_subtype, types=float),
                partial(check_value, reference=0, sign='>=', is_vector=True),
                partial(check_value, reference=1, sign='<=', is_vector=True)),
            'max_features': (
                partial(check_type, types=list),
                partial(check_subtype, types=int),
                partial(check_value, reference=0, sign='>', is_vector=True)),
            'class_weight': (
                partial(check_type, types=list),
                partial(check_value_in_set, options=(None, 'balanced'))),
            'ccp_alpha': (
                partial(check_type, types=list),
                partial(check_length, reference=2, sign='=='),
                partial(check_subtype, types=float),
                partial(check_value, reference=0, sign='>=', is_vector=True),
                partial(check_value, reference=1, sign='<=', is_vector=True)),
            'n_neighbors': (
                partial(check_type, types=list),
                partial(check_subtype, types=int),
                partial(check_value, reference=0, sign='>', is_vector=True)),
            'weights': (
                partial(check_type, types=list),
                partial(check_value_in_set, options=('distance', 'uniform'))),
            'leaf_size': (
                partial(check_type, types=list),
                partial(check_subtype, types=int),
                partial(check_value, reference=0, sign='>', is_vector=True)),
            'p': (
                partial(check_type, types=list),
                partial(check_subtype, types=int),
                partial(check_value, reference=0, sign='>', is_vector=True)),
            'C': (
                partial(check_type, types=list),
                partial(check_length, reference=2, sign='=='),
                partial(check_subtype, types=(int, float)),
                partial(check_value, reference=0, sign='>', is_vector=True)),
            'penalty': (
                partial(check_type, types=list),
                partial(check_value_in_set, options=(
                    'l1', 'l2', 'elasticnet'))),
            'tol': (
                partial(check_type, types=list),
                partial(check_subtype, types=(int, float)),
                partial(check_value, reference=0, sign='>', is_vector=True)),
            'priors': (
                partial(check_type, types=list),
                partial(check_subtype, types=(list, type(None)))),
            'var_smoothing': (
                partial(check_type, types=list),
                partial(check_length, reference=2, sign='=='),
                partial(check_subtype, types=(int, float)),
                partial(check_value, reference=0, sign='>', is_vector=True)),
            'input_neuron_number': (
                partial(check_type, types=list),
                partial(check_subtype, types=int),
                partial(check_value, reference=0, sign='>', is_vector=True)),
            'input_activation': (
                partial(check_type, types=list),
                partial(check_value_in_set, options=(
                    'elu', 'gelu', 'leaky_relu', 'linear', 'relu', 'softmax',
                    'softplus', 'swish'))),
            'hidden_neuron_number': (
                partial(check_type, types=list),
                partial(check_subtype, types=int),
                partial(check_value, reference=0, sign='>', is_vector=True)),
            'hidden_activation': (
                partial(check_type, types=list),
                partial(check_value_in_set, options=(
                    'elu', 'gelu', 'leaky_relu', 'linear', 'relu', 'softmax',
                    'softplus', 'swish'))),
            'input_dropout_rate': (
                partial(check_type, types=list),
                partial(check_subtype, types=(int, float)),
                partial(check_value, reference=0, sign='>=', is_vector=True)),
            'hidden_dropout_rate': (
                partial(check_type, types=list),
                partial(check_subtype, types=(int, float)),
                partial(check_value, reference=0, sign='>=', is_vector=True)),
            'batch_size': (
                partial(check_type, types=list),
                partial(check_subtype, types=int),
                partial(check_value, reference=0, sign='>', is_vector=True)),
            'learning_rate': (
                partial(check_type, types=list),
                partial(check_length, reference=2, sign='=='),
                partial(check_subtype, types=(int, float)),
                partial(check_value, reference=0, sign='>', is_vector=True)),
            'optimizer': (
                partial(check_type, types=list),
                partial(check_value_in_set, options=tuple(maps.NN_OPTS))),
            'loss': (
                partial(check_type, types=list),
                partial(check_value_in_set, options=tuple(maps.NN_LOSSES))),
            'n_estimators': (
                partial(check_type, types=list),
                partial(check_subtype, types=int),
                partial(check_value, reference=0, sign='>', is_vector=True)),
            'bootstrap': (
                partial(check_type, types=list),
                partial(check_value_in_set, options=(False, True))),
            'warm_start': (
                partial(check_type, types=list),
                partial(check_value_in_set, options=(False, True))),
            'kernel': (
                partial(check_type, types=list),
                partial(check_value_in_set, options=(
                    'linear', 'rbf', 'poly', 'sigmoid'))),
            'degree': (
                partial(check_type, types=list),
                partial(check_subtype, types=int),
                partial(check_value, reference=0, sign='>', is_vector=True)),
            'gamma': (
                partial(check_type, types=list),
                partial(check_length, reference=2, sign='=='),
                partial(check_subtype, types=(int, float)),
                partial(check_value, reference=0, sign='>', is_vector=True)),
            'graphs': (
                partial(check_type, types=list),
                partial(check_value_in_set, options=(
                    'AUC-ROC', 'AUC-PR', 'F1'))),
            'kpis': (
                partial(check_type, types=list),
                partial(check_value_in_set, options=(
                    'Logloss', 'Brier score', 'Subset accuracy',
                    'Cohen Kappa', 'Hamming loss', 'Jaccard score',
                    'Precision', 'Recall', 'F1 score', 'MCC', 'AUC')))}

        # Check if the data path is None
        if inputs['data_path'] is None:

            # Reduce the check map for the data path
            check_map['data_path'] = (check_map['data_path'][0],)

        # Check if the data columns are None
        if inputs['data_columns'] is None:

            # Reduce the check map for the data columns
            check_map['data_columns'] = (check_map['data_columns'][0],)

        # Loop over the dictionary items
        for key, value in inputs.items():

            # Loop over the check functions
            for function in check_map[key]:

                # Run the check function
                function(key, value)

            # Check if the key is 'data_column' and not None
            if key == 'data_columns' and value is not None:

                # Check if no feature has been passed
                if sum(type(item).__name__ in (
                        'DynamicFeature', 'StaticFeature')
                        for item in inputs['data_columns']) == 0:

                    # Raise an error to indicate missing features
                    raise ValueError(
                        "The internal model parameter 'data_columns' does not "
                        "contain at least one item of type 'DynamicFeature' "
                        "or 'StaticFeature'!")

                # Check if not exactly one label has been passed
                if sum(type(item).__name__ == 'Label'
                       for item in inputs['data_columns']) != 1:

                    # Raise an error to indicate a non-unique label
                    raise ValueError(
                        "The internal model parameter 'data_columns' does not "
                        "contain exactly one item of type 'Label'!")
