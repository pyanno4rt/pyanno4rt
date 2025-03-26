"""Machine learning component template."""

# Author: Tim Ortkamp

# %% External package import

from abc import ABCMeta, abstractmethod
from functools import partial
from os.path import abspath

# %% Internal package import

from pyanno4rt.checking import (
    check_key_in_dict, check_length, check_path, check_regular_extension,
    check_regular_extension_directory, check_subtype, check_type,
    check_value, check_value_in_set)
import pyanno4rt.learning._maps as maps
from pyanno4rt.tools import compare_dictionaries, filter_dict

# %% Class definition


class MachineLearningComponent(metaclass=ABCMeta):
    """
    Machine learning component template class.

    Parameters
    ----------
    name : str
        Name of the component class.

    segment : str
        Name of the segment associated with the component.

    component_type : {'constraint', 'objective'}
        Type of the component.

    parameter_name : tuple
        Name of the component parameters.

    parameter_category : tuple
        Category of the component parameters.

    model_parameters : dict
        Dictionary with the data handling & learning model parameters:

        - model_label : str
            Label for the machine learning model.

        - model_folder_path : None or str, default=None
            Path to a folder for loading an external machine learning model.

        - data_path : str
            Path to the data set used for fitting the machine learning model.

        - data_columns : list
            List of \
            :class:`~pyanno4rt.learning.features._columns.DynamicFeature` \
            or :class:`~pyanno4rt.learning.features._columns.StaticFeature` \
            and :class:`~pyanno4rt.learning.features._columns.Label` objects.

        - preprocessing_steps : list, default=['Identity']
            Sequence of labels associated with preprocessing algorithms to \
            preprocess the input features.

            The following preprocessing steps are currently available:

            - 'Identity' \
                :class:`~pyanno4rt.learning.preprocessing._identity.Identity`
            - 'StandardScaler' \
                :class:`~pyanno4rt.learning.preprocessing._standard_scaler.StandardScaler`
            - 'Whitening' \
                :class:`~pyanno4rt.learning.preprocessing._whitening.Whitening`

        - architecture : {'vanilla-input-convex', 'vanilla'}, default='vanilla'
            Type of architecture for the neural network model.

        - max_hidden_layers : int, default=2
            Maximum number of hidden layers for the neural network model.

        - tune_space : dict, default={}
            Search space for the Bayesian hyperparameter optimization.

        - tune_evaluations : int, default=50
            Number of evaluation steps (trials) for the Bayesian \
            hyperparameter optimization.

        - tune_score : {'AUC', 'Brier score', 'Logloss'}, default='Logloss'
            Scoring function for the evaluation of the hyperparameter set \
            candidates.

        - tune_splits : int, default=5
            Number of splits for the stratified cross-validation within each \
            hyperparameter optimization step.

        - tune_repeats : int, default=1
            Number of repeats for the stratified cross-validation within each \
            hyperparameter optimization step.

        - inspect_model : bool, default=False
            Indicator for the inspection of the machine learning model.

        - evaluate_model : bool, default=False
            Indicator for the evaluation of the machine learning model.

        - oof_splits : int, default=5
            Number of splits for the stratified cross-validation within the \
            out-of-folds evaluation step.

        - oof_repeats : int, default=1
            Number of repeats for the stratified cross-validation within the \
            out-of-folds evaluation step.

        - write_features : bool, default=False
            Indicator for writing the iteratively calculated feature vectors \
            into a feature history.

        - display_options : dict, \
            default={'graphs': ['AUC-ROC', 'AUC-PR', 'F1'],\
                     'kpis': ['Logloss', 'Brier score', 'Subset accuracy', \
                              'Cohen Kappa', 'Hamming loss', 'Jaccard score', \
                              'Precision', 'Recall', 'F1 score', 'MCC', 'AUC']}
            Dictionary with the graph and KPI display options.

    embedding : {'active', 'passive'}
        Mode of embedding for the component. In 'passive' mode, the component \
        value is computed and tracked, but not considered in the optimization \
        problem, unlike in 'active' mode.

    weight : int or float
        Weight of the component function.

    rank : int, default=1
        Rank of the component in the lexicographic order.

    bounds : None or list
        Constraint bounds for the component.

    link : None or list
        Other segments used for joint evaluation.

    identifier : None or str
        Additional string for naming the component.

    display : bool
        Indicator for the display of the component.

    Attributes
    ----------
    name : str
        See 'Parameters'.

    segment : str
        See 'Parameters'.

    component_type : {'constraint', 'objective'}
        See 'Parameters'.

    parameter_name : tuple
        See 'Parameters'.

    parameter_category : tuple
        See 'Parameters'.

    parameter_value : list
        Value of the component parameters.

    model_parameters : dict
        See 'Parameters'.

    embedding : {'active', 'passive'}
        See 'Parameters'.

    weight : float
        See 'Parameters'.

    rank : int
        See 'Parameters'.

    bounds : list
        See 'Parameters'.

    link : None or list
        See 'Parameters'.

    identifier : None or str
        See 'Parameters'.

    display : bool
        See 'Parameters'.

    data_model_handler : None
        Initial variable for the object used to handle the dataset, the \
        feature map generation and the feature (re-)calculation.

    model : None
        Initial variable for the object used to preprocess, tune, train, \
        inspect and evaluate the machine learning model.

    adjusted_parameters : bool
        Indicator for the adjustment of the parameters due to fractionation.
    """

    def __init__(
            self,
            name,
            segment,
            component_type,
            parameter_name,
            parameter_category,
            model_parameters,
            embedding,
            weight,
            rank,
            bounds,
            link,
            identifier,
            display):

        # Check the input arguments
        self.check(
            filter_dict(locals(), remove_keys=('self',))
            | model_parameters
            | model_parameters['tune_space']
            | model_parameters['display_options'])

        # Set the instance attributes from the class arguments
        self.name = name
        self.segment = segment
        self.component_type = component_type
        self.parameter_name = parameter_name
        self.parameter_category = parameter_category
        self.parameter_value = []
        self.model_parameters = {
            'model_label': model_parameters.get('model_label'),
            'model_folder_path': model_parameters.get('model_folder_path'),
            'data_path': model_parameters.get('data_path'),
            'data_columns': model_parameters.get('data_columns'),
            'preprocessing_steps': model_parameters.get(
                'preprocessing_steps', ['Identity']),
            'architecture': model_parameters.get('architecture', 'vanilla'),
            'max_hidden_layers': model_parameters.get('max_hidden_layers', 2),
            'tune_space': model_parameters.get('tune_space', {}),
            'tune_evaluations': model_parameters.get('tune_evaluations', 50),
            'tune_score': model_parameters.get('tune_score', 'Logloss'),
            'tune_splits': model_parameters.get('tune_splits', 5),
            'tune_repeats': model_parameters.get('tune_repeats', 1),
            'inspect_model': model_parameters.get('inspect_model', False),
            'evaluate_model': model_parameters.get('evaluate_model', False),
            'oof_splits': model_parameters.get('oof_splits', 5),
            'oof_repeats': model_parameters.get('oof_repeats', 1),
            'write_features': model_parameters.get('write_features', False),
            'display_options': model_parameters.get(
                'display_options', {
                    'graphs': ['AUC-ROC', 'AUC-PR', 'F1'],
                    'kpis': [
                        'Logloss', 'Brier score', 'Subset accuracy',
                        'Cohen Kappa', 'Hamming loss', 'Jaccard score',
                        'Precision', 'Recall', 'F1 score', 'MCC', 'AUC']})
            }
        self.embedding = embedding
        self.weight = float(weight)
        self.rank = rank
        self.bounds = (
            (0.0, 1.0) if bounds is None or embedding == 'passive' else (
                0.0 if bounds[0] is None else float(bounds[0]),
                1.0 if bounds[1] is None else float(bounds[1])))
        self.link = [] if link is None else link
        self.identifier = identifier
        self.display = display

        # Check if the model folder path is not None
        if model_parameters['model_folder_path'] is not None:

            # Convert the model folder path into the absolute value
            model_parameters['model_folder_path'] = abspath(
                model_parameters['model_folder_path'])

        # Check if the data path is not None
        if model_parameters['data_path'] is not None:

            # Convert the data path into the absolute value
            model_parameters['data_path'] = abspath(
                model_parameters['data_path'])

        # Initialize the data model handler and the outcome model
        self.data_model_handler = None
        self.model = None

        # Initialize the adjustment indicator
        self.adjusted_parameters = False

    def __eq__(
            self,
            other):
        """
        Compare an instance with another object.

        Parameters
        ----------
        other : object
            The object to compare the instance with.

        Returns
        -------
        bool
            Indicator for the equality of the objects.
        """

        return (
            all(self.__dict__[key] == other.__dict__[key] for key in (
                'name', 'segment', 'component_type', 'link', 'identifier'))
            and compare_dictionaries(
                self.__dict__.get('model_parameters', {}),
                other.__dict__.get('model_parameters', {})))

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

        # Get the check map
        check_map = {
            'name': (
                partial(check_type, types=str),),
            'segment': (
                partial(check_type, types=str),),
            'component_type': (
                partial(check_type, types=str),
                partial(
                    check_value_in_set, options=('constraint', 'objective'))),
            'parameter_name': (
                partial(check_type, types=tuple),
                partial(check_subtype, types=str)),
            'parameter_category': (
                partial(check_type, types=tuple),
                partial(check_subtype, types=str)),
            'model_parameters': (
                partial(check_type, types=(type(None), dict)),),
            'embedding': (
                partial(check_type, types=str),
                partial(check_value_in_set, options=('active', 'passive'))),
            'weight': (
                partial(check_type, types=(int, float)),
                partial(check_value, reference=0, sign='>')),
            'rank': (
                partial(check_type, types=int),
                partial(check_value, reference=0, sign='>')),
            'bounds': (
                partial(check_type, types=(type(None), list)),
                partial(check_length, reference=2, sign='=='),
                partial(check_subtype, types=(type(None), int, float))),
            'link': (
                partial(check_type, types=(type(None), list)),
                partial(check_subtype, types=str)),
            'identifier': (
                partial(check_type, types=(type(None), str)),),
            'display': (
                partial(check_type, types=bool),),
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
            'preprocessing_steps': (
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
            'inspect_model': (
                partial(check_type, types=bool),),
            'evaluate_model': (
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

        # Loop over the dictionary keys
        for key, value in inputs.items():

            # Loop over the check functions
            for function in check_map[key]:

                # Run the check function
                function(key, value)

            # Check if the key is 'data_column' and not None
            if key == 'data_columns' and value is not None:

                # Check if no feature has been passed
                if sum([type(item).__name__ in (
                        'DynamicFeature', 'StaticFeature')
                        for item in inputs['data_columns']]) == 0:

                    # Raise an error to indicate missing features
                    raise ValueError(
                        "The treatment plan parameter 'data_columns' does not "
                        "contain at least one item of type 'DynamicFeature' "
                        "or 'StaticFeature'!")

                # Check if not exactly one label has been passed
                if sum([type(item).__name__ == 'Label'
                        for item in inputs['data_columns']]) != 1:

                    # Raise an error to indicate a non-unique label
                    raise ValueError(
                        "The treatment plan parameter 'data_columns' does not "
                        "contain exactly one item of type 'Label'!")

    def get_class(self):
        """
        Get the name of the component class.

        Returns
        -------
        str
            Name of the component class.
        """

        return 'MachineLearningComponent'

    def get_parameter_value(self):
        """
        Get the value of the parameters.

        Returns
        -------
        list
            Value of the parameters.
        """

        return self.parameter_value

    def set_parameter_value(
            self,
            value):
        """
        Set the value of the parameters.

        Parameters
        ----------
        value : list
            Value to be set.
        """

        self.parameter_value = value

    def get_weight_value(self):
        """
        Get the value of the weight.

        Returns
        -------
        float
            Value of the weight.
        """

        return self.weight

    def set_weight_value(
           self,
           value):
        """
        Set the value of the weight.

        Parameters
        ----------
        value : float
            Value to be set.
        """

        self.weight = value

    @abstractmethod
    def add_model(self):
        """Add the machine learning model to the component."""

    @abstractmethod
    def compute_value(
            self,
            dose,
            segment):
        """Compute the component value."""

    @abstractmethod
    def compute_gradient(
            self,
            dose,
            segment):
        """Compute the component gradient."""
