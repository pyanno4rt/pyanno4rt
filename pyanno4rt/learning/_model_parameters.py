"""Model parameters."""

# Author: Tim Ortkamp

# %% External package import

from copy import deepcopy
from functools import partial
from os.path import abspath

# %% Internal package import

from pyanno4rt.learning.evaluation import DisplayOptions
import pyanno4rt.learning._maps as maps
from pyanno4rt.tools import filter_dict
from pyanno4rt.validation import (
    validate_directory, validate_file, validate_length, validate_path,
    validate_subtype, validate_type, validate_value, validate_value_in_set)

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

    model_type : {'forest', 'logistic', 'naive_bayes', 'neighbors', \
                  'neural_network', 'svm', 'tree'}
        Type of the machine learning model.

    model_folder_path : None or str
        Path to an external model folder.

    data_path : None or str
        Path to the data set used for fitting the learning model.

    data_columns : None or list, default=None
        List of \
        :class:`~pyanno4rt.learning.features._columns.DynamicFeature` \
        or :class:`~pyanno4rt.learning.features._columns.StaticFeature` \
        and :class:`~pyanno4rt.learning.features._columns.Label` objects.

    preprocessing : None or list, default=None
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

    tune_space : None or object, default=None
        The object used to represent the search space for the Bayesian \
        hyperparameter optimization, see the classes \
            :class:`~pyanno4rt.learning.forest._tune_space_rf.TuneSpaceRF`\
            :class:`~pyanno4rt.learning.logistic._tune_space_lr.TuneSpaceLR`\
            :class:`~pyanno4rt.learning.naive_bayes._tune_space_nb.TuneSpaceNB`\
            :class:`~pyanno4rt.learning.neighbors._tune_space_knn.TuneSpaceKNN`\
            :class:`~pyanno4rt.learning.neural_network._tune_space_nn.TuneSpaceNN`\
            :class:`~pyanno4rt.learning.svm._tune_space_svm.TuneSpaceSVM`\
            :class:`~pyanno4rt.learning.tree._tune_space_dt.TuneSpaceDT`.

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

    display_options : object of class \
        :class:`~pyanno4rt.learning.evaluation._display_options_DisplayOptions`,\
        default=DisplayOptions()
        The object used to represent the graph and KPI display options.

    Attributes
    ----------
    model_label : str
        See 'Parameters'.

    model_type : {'forest', 'logistic', 'naive_bayes', 'neighbors', \
                  'neural_network', 'svm', 'tree'}
        See 'Parameters'.

    model_folder_path : None or str
        See 'Parameters'.

    data_path : None or str
        See 'Parameters'.

    data_columns : None or list
        See 'Parameters'.

    preprocessing : None or list
        See 'Parameters'.

    architecture : {'vanilla-input-convex', 'vanilla'}
        See 'Parameters'.

    max_hidden_layers : int
        See 'Parameters'.

    tune_space : None or object
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

    display_options : object of class \
        :class:`~pyanno4rt.learning.evaluation._display_options_DisplayOptions`
        See 'Parameters'.
    """

    def __init__(
            self,
            model_label,
            model_type,
            model_folder_path=None,
            data_path=None,
            data_columns=None,
            preprocessing=None,
            architecture='vanilla',
            max_hidden_layers=2,
            tune_space=None,
            tune_evaluations=50,
            tune_score='Logloss',
            tune_splits=5,
            tune_repeats=1,
            inspect=False,
            evaluate=False,
            oof_splits=5,
            oof_repeats=1,
            write_features=False,
            display_options=DisplayOptions()):

        # Check if the data columns are None
        if data_columns is None:

            # Get the default data columns
            data_columns = []

        # Check if the preprocessing is None
        if preprocessing is None:

            # Get the default preprocessing
            preprocessing = ['Identity']

        # Check if the tune space is None
        if tune_space is None:

            # Get the default tune space
            tune_space = maps.SPACES[model_type]()

        # Get the input arguments
        self.inputs = filter_dict(vars(), remove_keys=('self',))

        # Check if a model folder path has been passed
        if self.inputs['model_folder_path'] is not None:

            # Convert the model folder path into an absolute path
            self.inputs['model_folder_path'] = abspath(
                self.inputs['model_folder_path'])

        # Check if a data path has been passed
        if self.inputs['data_path'] is not None:

            # Convert the data path into an absolute path
            self.inputs['data_path'] = abspath(self.inputs['data_path'])

        # Check the input arguments
        self.validate(self.inputs)

        # Loop over the input arguments
        for item in self.inputs.items():

            # Set the attribute
            setattr(self, *item)

    def to_dict(self):
        """Serialize the model parameters into a dictionary."""

        # Get the parameter dictionary
        dictionary = deepcopy(self.inputs)

        # Serialize the data columns
        dictionary['data_columns'] = [
            item.to_dict() for item in dictionary['data_columns']]

        # Serialize the tune space
        dictionary['tune_space'] = self.tune_space.to_dict()

        # Serialize the display options
        dictionary['display_options'] = self.display_options.to_dict()

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

        # Deserialize the tune space
        dictionary['tune_space'] = maps.SPACES[dictionary['model_type']](
            **dictionary['tune_space'])

        # Deserialize the display options
        dictionary['display_options'] = DisplayOptions(
            **dictionary['display_options'])

        return cls(**dictionary)

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
            'model_label': (
                partial(validate_type, options=str),),
            'model_type': (
                partial(validate_type, options=str),
                partial(validate_value_in_set, options=(
                    'forest', 'logistic', 'naive_bayes', 'neighbors',
                    'neural_network', 'svm', 'tree'))),
            'model_folder_path': (
                partial(validate_type, options=(type(None), str)),
                partial(validate_path)),
            'data_path': (
                partial(validate_type, options={
                    True: (type(None), str), False: str},
                    type_condition=isinstance(
                        inputs.get('model_folder_path'), str)),
                partial(validate_file, options=('.csv',)),
                partial(validate_directory, options=(
                    '.jpg', '.npy', '.npz', '.png'), alt=('.csv',))),
            'data_columns': (
                partial(validate_type, options={
                    True: (type(None), list), False: list},
                    type_condition=isinstance(
                        inputs.get('model_folder_path'), str)),
                partial(validate_length, reference=2, sign='>=')),
            'preprocessing': (
                partial(validate_type, options=list),
                partial(validate_subtype, options=str),
                partial(
                    validate_value_in_set, options=tuple(maps.TRANSFORMERS))),
            'architecture': (
                partial(validate_type, options=str),
                partial(validate_value_in_set, options=(
                    'vanilla', 'vanilla input-convex'))),
            'max_hidden_layers': (
                partial(validate_type, options=int),
                partial(validate_value, reference=1, sign='>=')),
            'tune_space': (
                partial(validate_type, options=tuple(maps.SPACES.values())),),
            'tune_evaluations': (
                partial(validate_type, options=int),
                partial(validate_value, reference=0, sign='>')),
            'tune_score': (
                partial(validate_type, options=str),
                partial(validate_value_in_set, options=tuple(
                    ('AUC', *maps.LOSSES)))),
            'tune_splits': (
                partial(validate_type, options=int),
                partial(validate_value, reference=1, sign='>=')),
            'tune_repeats': (
                partial(validate_type, options=int),
                partial(validate_value, reference=1, sign='>=')),
            'inspect': (
                partial(validate_type, options=bool),),
            'evaluate': (
                partial(validate_type, options=bool),),
            'oof_splits': (
                partial(validate_type, options=int),
                partial(validate_value, reference=1, sign='>=')),
            'oof_repeats': (
                partial(validate_type, options=int),
                partial(validate_value, reference=1, sign='>=')),
            'write_features': (
                partial(validate_type, options=bool),),
            'display_options': (
                partial(validate_type, options=DisplayOptions),)}

        # Check if the data path is None
        if inputs['data_path'] is None:

            # Reduce the validation map for the data path
            validation_map['data_path'] = (validation_map['data_path'][0],)

        # Check if the data columns are None
        if inputs['data_columns'] is None:

            # Reduce the validation map for the data columns
            validation_map['data_columns'] = (
                validation_map['data_columns'][0],)

        # Loop over the dictionary items
        for key, value in inputs.items():

            # Loop over the validation functions
            for function in validation_map[key]:

                # Run the validation function
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
