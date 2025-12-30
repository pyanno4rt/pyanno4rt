"""Random forest model."""

# Author: Tim Ortkamp

# %% External package import

from os.path import abspath
from pickle import dump, load

from copy import deepcopy
from functools import partial
from glob import glob
from numpy import ones
from sklearn.ensemble import RandomForestClassifier

# %% Internal package import

from pyanno4rt.learning.datasets import TabularDataset
from pyanno4rt.learning.evaluation import ModelEvaluator
from pyanno4rt.learning.inspection import ModelInspector
from pyanno4rt.learning._maps import TUNERS
from pyanno4rt.learning.preprocessing import TabularPreprocessor
from pyanno4rt.logging import get_logger
from pyanno4rt.tools import filter_dict
from pyanno4rt.validation import validate_path, validate_type

# %% Class definition


class RandomForest():
    """
    Random forest model class.

    This class implements methods to handle random forest models.

    Parameters
    ----------
    label : str
        Label for the learning model.

    dataset : object of class \
        :class:`~pyanno4rt.learning.datasets._tabular_dataset.TabularDataset`
        The object used to represent the dataset.

    preprocessor : None or object of class \
        :class:`~pyanno4rt.learning.preprocessing._tabular_preprocessor.TabularPreprocessor`,\
        default=None
        The object used to represent the data preprocessor.

    tuner : None or object of class \
        :class:`~pyanno4rt.learning.tuning._bayes_hp_tuner.BayesHPTuner`\
        :class:`~pyanno4rt.learning.tuning._grid_hp_tuner.GridHPTuner`\
        :class:`~pyanno4rt.learning.tuning._random_hp_tuner.RandomHPTuner`,\
        default=None
        The object used to represent the hyperparameter tuner.

    inspector : None or object of class \
        :class:`~pyanno4rt.learning.inspection._model_inspector.ModelInspector`,\
        default=None
        The object used to represent the model inspector.

    evaluator : None or object of class \
        :class:`~pyanno4rt.learning.evaluation._model_evaluator.ModelEvaluator`,\
        default=None
        The object used to represent the model evaluator.

    model_path : None or str, default=None
        Path to an external model.

    Attributes
    ----------
    arguments : dict
        Dictionary with the model input arguments (for serialization).

    label : str
        See 'Parameters'.

    dataset : object of class \
        :class:`~pyanno4rt.learning.datasets._tabular_dataset.TabularDataset`
        See 'Parameters'.

    preprocessor : None or object of class \
        :class:`~pyanno4rt.learning.preprocessing._tabular_preprocessor.TabularPreprocessor`
        See 'Parameters'.

    tuner : None or object of class \
        :class:`~pyanno4rt.learning.tuning._bayes_hp_tuner.BayesHPTuner`\
        :class:`~pyanno4rt.learning.tuning._grid_hp_tuner.GridHPTuner`\
        :class:`~pyanno4rt.learning.tuning._random_hp_tuner.RandomHPTuner`
        See 'Parameters'.

    inspector : None or object of class \
        :class:`~pyanno4rt.learning.inspection._model_inspector.ModelInspector`
        See 'Parameters'.

    evaluator : None or object of class \
        :class:`~pyanno4rt.learning.evaluation._model_evaluator.ModelEvaluator`
        See 'Parameters'.

    model_path : None or str
        See 'Parameters'.

    hyperparameters : dict
        Dictionary with the model hyperparameters.

    predictor : object of class \
        :class:`~sklearn.ensemble.RandomForestClassifier`
        The object used to represent the prediction model.

    _reload_data : bool
        Indicator for updating the dataset.

    _reset_calc : bool
        Indicator for updating the feature calculator.

    _refit : bool
        Indicator for updating the model.

    _reinspect : bool
        Indicator for updating the model inspections.

    _reevaluate : bool
        Indicator for updating the model evaluations.
    """

    def __init__(
            self,
            label,
            dataset,
            preprocessor=None,
            tuner=None,
            inspector=None,
            evaluator=None,
            model_path=None):

        # Check if a path has been provided
        if model_path is not None:

            # Convert the path into an absolute value
            model_path = abspath(model_path)

        # Get the input arguments
        self.arguments = filter_dict(vars(), remove_keys=('self',))

        # Check the input arguments
        self.validate(self.arguments)

        # Get the instance attributes
        self.label = label
        self.dataset = dataset
        self.preprocessor = preprocessor
        self.tuner = tuner
        self.inspector = inspector
        self.evaluator = evaluator
        self.model_path = model_path

        # Initialize the prediction model attributes
        self.feature_calculator = None
        self.hyperparameters = {
            'n_estimators': 100,
            'criterion': 'gini',
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'min_weight_fraction_leaf': 0.0,
            'max_features': 'sqrt',
            'max_leaf_nodes': None,
            'min_impurity_decrease': 0.0,
            'bootstrap': True,
            'oob_score': False,
            'n_jobs': -1,
            'random_state': 12,
            'verbose': 0,
            'warm_start': False,
            'class_weight': None,
            'ccp_alpha': 0.0,
            'max_samples': None,
            'monotonic_cst': None}
        self.predictor = RandomForestClassifier(**self.hyperparameters)

        # Initialize the refreshing indicators
        self._reload_data = True
        self._reset_calc = True
        self._refit = True
        self._reinspect = True
        self._reevaluate = True

    def to_dict(self):
        """Serialize the model into a dictionary."""

        # Get the parameter dictionary
        dictionary = deepcopy(self.arguments)

        # Loop over the object-related keys
        for key in (
                'dataset', 'preprocessor', 'tuner', 'inspector', 'evaluator'):

            # Check if a value is available
            if dictionary[key] is not None:

                # Serialize the attribute objects
                dictionary[key] = dictionary[key].to_dict()

        return dictionary

    @classmethod
    def from_dict(
            cls,
            dictionary):
        """
        Deserialize the model from a dictionary.

        Parameters
        ----------
        dictionary : dict
            Dictionary with the model parameters.

        Returns
        -------
        object of class \
            :class:`~pyanno4rt.learning.models.forest._random_forest.RandomForest`
            The object used to represent the model.
        """

        # Deserialize the dataset
        dictionary['dataset'] = TabularDataset.from_dict(dictionary['dataset'])

        # Check if a preprocessor dictionary has been passed
        if dictionary['preprocessor'] is not None:

            # Deserialize the preprocessor
            dictionary['preprocessor'] = TabularPreprocessor.from_dict(
                dictionary['preprocessor'])

        # Check if a tuner dictionary has been passed
        if dictionary['tuner'] is not None:

            # Deserialize the tuner
            dictionary['tuner'] = (
                TUNERS[dictionary['tuner'].pop('name')].from_dict(
                    dictionary['tuner']))

        # Check if an inspector dictionary has been passed
        if dictionary['inspector'] is not None:

            # Deserialize the inspector
            dictionary['inspector'] = ModelInspector.from_dict(
                dictionary['inspector'])

        # Check if an evaluator dictionary has been passed
        if dictionary['evaluator'] is not None:

            # Deserialize the evaluator
            dictionary['evaluator'] = ModelEvaluator.from_dict(
                dictionary['evaluator'])

        return cls(**dictionary)

    def load_data(self):
        """Load the dataset."""

        # Log a message about loading the dataset
        get_logger().info("Loading dataset for '%s' ...", self.label)

        # Load the dataset
        self.dataset.load()

        # Generate the data
        self.dataset.generate()

    def add_calculator(
            self,
            calculator):
        """
        Add the feature calculator.

        Parameters
        ----------
        calculator : object of class \
            :class:`~pyanno4rt.learning.features._feature_calculator.FeatureCalculator`
            The object used to (re)calculate input features and gradients.
        """

        # Log a message about adding the feature calculator
        get_logger().info("Adding feature calculator for '%s' ...", self.label)

        # Initialize the feature calculator
        self.feature_calculator = calculator

        # Add the feature map
        self.feature_calculator.set_mapping(self.dataset.feature_map)

    def fit_preprocessor(
            self,
            features,
            labels=None):
        """
        Fit the preprocessor.

        Parameters
        ----------
        features : ndarray
            Values of the input features.

        labels : ndarray, default=None
            Values of the input labels.
        """

        # Check if a preprocessor has been provided
        if self.preprocessor is not None:

            # Fit the preprocessor and transform the data
            self.preprocessor.fit(features, labels)

    def preprocess(
            self,
            features,
            labels=None):
        """
        Preprocess the inputs.

        Parameters
        ----------
        features : ndarray
            Values of the input features.

        labels : ndarray, default=None
            Values of the input labels.
        """

        # Check if a preprocessor has been provided
        if self.preprocessor is not None:

            # Transform the data
            return self.preprocessor.transform(features, labels)

        return features, labels

    def tune_hyperparameters(
            self,
            features,
            labels):
        """
        Tune the model hyperparameters.

        Parameters
        ----------
        features : ndarray
            Values of the input features.

        labels : ndarray
            Values of the input labels.
        """

        # Check if a tuner has been provided
        if self.tuner is not None:

            # Search the hyperparameter set
            proposal = self.tuner.search(
                deepcopy(self), features, labels, self.dataset.folds)

            # Check if a Bayesian hyperparameter tuner has been provided
            if isinstance(self.tuner, TUNERS['Bayes']):

                # Get the full hyperparameter set
                self.get_bayes_hp(proposal)

            # Check if a grid hyperparameter tuner has been provided
            elif isinstance(self.tuner, TUNERS['Grid']):

                # Get the full hyperparameter set
                self.get_grid_hp(proposal)

            # Check if a random hyperparameter tuner has been provided
            elif isinstance(self.tuner, TUNERS['Random']):

                # Get the full hyperparameter set
                self.get_random_hp(proposal)

    def get_bayes_hp(
            self,
            proposal):
        """
        Get the hyperparameters from a Bayesian search proposal set.

        Parameters
        ----------
        proposal : dict
            Proposal for the tunable hyperparameters.
        """

        # Build the hyperparameter dictionary
        self.hyperparameters = self.hyperparameters | {
            'n_estimators': proposal.get('n_estimators', 100),
            'criterion': proposal.get('criterion', 'gini'),
            'max_depth': proposal.get('max_depth'),
            'min_samples_split': proposal.get('min_samples_split', 2),
            'min_samples_leaf': proposal.get('min_samples_leaf', 1),
            'min_weight_fraction_leaf': proposal.get(
                'min_weight_fraction_leaf', 0.0),
            'max_features': proposal.get('max_features', 'sqrt'),
            'bootstrap': proposal.get('bootstrap', True),
            'class_weight': proposal.get('class_weight'),
            'ccp_alpha': proposal.get('ccp_alpha', 0.0)}

    def get_grid_hp(
            self,
            proposal):
        """
        Get the hyperparameters from a grid search proposal set.

        Parameters
        ----------
        proposal : dict
            Proposal for the tunable hyperparameters.
        """

    def get_random_hp(
            self,
            proposal):
        """
        Get the hyperparameters from a random search proposal set.

        Parameters
        ----------
        proposal : dict
            Proposal for the tunable hyperparameters.
        """

    def fit_predictor(
            self,
            features,
            labels):
        """
        Fit the model.

        Parameters
        ----------
        features : ndarray
            Values of the input features.

        labels : ndarray
            Values of the input labels.
        """

        # Initialize the predictor
        self.predictor = self.predictor.set_params(**self.hyperparameters)

        # Fit the predictor
        self.predictor.fit(features, labels)

    def predict(
            self,
            features):
        """
        Predict the label values.

        Parameters
        ----------
        features : ndarray
            Values of the input features.

        Returns
        -------
        float or ndarray
            Value(s) of the predicted label(s).
        """

        # Check if the feature array has only a single row
        if features.shape[0] == 1:

            # Return a single label prediction value
            return self.predictor.predict_proba(features)[0][1]

        # Else, return an array with label predictions
        return self.predictor.predict_proba(features)[:, 1]

    def inspect(self):
        """Inspect the model."""

        # Check if an inspector has been provided
        if self.inspector is not None:

            # Log a message about inspecting the model
            get_logger().info("Inspecting model '%s' ...", self.label)

            # Run the model inspection
            self.inspector.run(deepcopy(self))

    def evaluate(self):
        """Evaluate the model."""

        # Check if an evaluator has been provided
        if self.evaluator is not None:

            # Log a message about evaluating the model
            get_logger().info("Evaluating model '%s' ...", self.label)

            # Run the model evaluation
            self.evaluator.run(deepcopy(self))

    def featurize(
            self,
            dose,
            segment):
        """
        Compute the model input feature vector.

        Parameters
        ----------
        dose : tuple
            Dose vectors.

        segment : tuple
            Segment names.

        Returns
        -------
        ndarray
            Feature vector.
        """

        return self.feature_calculator.featurize(dose, segment)

    def gradientize(
            self,
            dose,
            segment):
        """
        Derive the model gradients.

        Parameters
        ----------
        dose : tuple
            Dose vectors.

        segment : tuple
            Segment names.

        Returns
        -------
        ndarray
            Feature gradient w.r.t dose.

        ndarray
            Preprocessor gradient w.r.t the features.

        ndarray
            Predictor gradient w.r.t the preprocessed features.
        """

        # Get the feature vector
        features = self.featurize(dose, segment)

        # Derive the feature gradient
        feature_gradient = self.feature_calculator.gradientize(dose, segment)

        # Check if a preprocessor object has been provided
        if self.preprocessor is not None:

            # Derive the preprocessing gradient
            preprocessing_gradient = self.preprocessor.gradientize(features)

        else:

            # Use the default preprocessing gradient
            preprocessing_gradient = ones((len(features),))

        # Derive the predictor gradient
        predictor_gradient = ones((len(features),))

        return feature_gradient, preprocessing_gradient, predictor_gradient

    def load(self):
        """Load an external model."""

        # Log a message about loading the model
        get_logger().info("Loading '%s' model from file ...", self.label)

        # Check if a preprocessor has been found
        if len(glob(self.model_path+'/preprocessor.sav')) == 1:

            # Open a file stream for the preprocessor
            with open(self.model_path+'/preprocessor.sav', 'rb') as file:

                # Load the preprocessor
                self.preprocessor = load(file)

        else:

            # Set the preprocessor to None
            self.preprocessor = None

        # Open a file stream for the predictor
        with open(self.model_path+'/predictor.sav', 'rb') as file:

            # Load the predictor
            self.predictor = load(file)

        # Get the hyperparameters
        self.hyperparameters = self.predictor.get_params()

    def save(
            self,
            path):
        """
        Save the model.

        Parameters
        ----------
        path : str
            Path for storing the model.
        """

        # Check if a preprocessor object exists
        if self.preprocessor is not None:

            # Open a file stream for the preprocessor
            with open(path+'/preprocessor.sav', 'wb') as file:

                # Dump the preprocessor
                dump(self.preprocessor, file)

        # Open a file stream for the predictor
        with open(path+'/predictor.sav', 'wb') as file:

            # Dump the predictor
            dump(self.predictor, file)

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
            'label': (
                partial(validate_type, options=str),
                ),
            'dataset': (
                partial(validate_type, options=TabularDataset),
                ),
            'preprocessor': (
                partial(validate_type, options=(
                    type(None), TabularPreprocessor)),
                ),
            'tuner': (
                partial(validate_type, options=(type(None), *TUNERS.values())),
                ),
            'inspector': (
                partial(validate_type, options=(type(None), ModelInspector)),
                ),
            'evaluator': (
                partial(validate_type, options=(type(None), ModelEvaluator)),
                ),
            'model_path': (
                partial(validate_type, options={
                    True: str,
                    False: (type(None), str)},
                    condition=(
                        inputs['dataset'] is None or
                        inputs['dataset'].data_path is None)),
                partial(validate_path)
                )
            }

        # Loop over the dictionary items
        for key, value in inputs.items():

            # Loop over the validation functions
            for function in validation_map[key]:

                # Run the validation function
                function(key, value)
