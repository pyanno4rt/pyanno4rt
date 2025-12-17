"""Logistic regression model."""

# Author: Tim Ortkamp

# %% External package import

from os.path import abspath
from pickle import dump, load

from copy import deepcopy
from functools import partial
from numpy import ones
from sklearn.linear_model import LogisticRegression as skLogReg

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


class LogisticRegression():
    """
    Logistic regression model class.

    This class implements methods to handle logistic regression models.

    Parameters
    ----------
    label : str
        Label for the learning model.

    dataset : object of class \
        :class:`~pyanno4rt.learning.datasets._tabular_dataset.TabularDataset`,
        default=None
        The object used to represent the dataset.

    preprocessor : None or object of class \
        :class:`~pyanno4rt.learning.preprocessing._tabular_preprocessor.TabularPreprocessor`,
        default=None
        The object used to represent the data preprocessor.

    tuner : None or object of class \
        :class:`~pyanno4rt.learning.tuning._bayes_hp_tuner.BayesHPTuner`\
        :class:`~pyanno4rt.learning.tuning._grid_hp_tuner.GridHPTuner`\
        :class:`~pyanno4rt.learning.tuning._random_hp_tuner.RandomHPTuner`,
        default=None
        The object used to represent the hyperparameter tuner.

    inspector : None or object of class \
        :class:`~pyanno4rt.learning.inspection._model_inspector.ModelInspector`,
        default=None
        The object used to represent the model inspector.

    evaluator : None or object of class \
        :class:`~pyanno4rt.learning.evaluation._model_evaluator.ModelEvaluator`,
        default=None
        The object used to represent the model evaluator.

    path : None or str, default=None
        Path to an external model folder.

    Attributes
    ----------
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

    path : None or str
        See 'Parameters'.

    hyperparameters : dict
        Dictionary with the model hyperparameters.

    predictor : object of class \
        :class:`~sklearn.linear_model.LogisticRegression`
        The object used to represent the prediction model.
    """

    def __init__(
            self,
            label,
            dataset,
            preprocessor=None,
            tuner=None,
            inspector=None,
            evaluator=None,
            path=None):

        # Check if a path has been provided
        if path is not None:

            # Convert the path into an absolute value
            path = abspath(path)

        # Get the input arguments
        self.inputs = filter_dict(vars(), remove_keys=('self',))

        # Check the input arguments
        self.validate(self.inputs)

        # Get the instance attributes
        self.label = label
        self.dataset = dataset
        self.preprocessor = preprocessor
        self.tuner = tuner
        self.inspector = inspector
        self.evaluator = evaluator
        self.path = path

        # Initialize the prediction model attributes
        self.feature_calculator = None
        self.hyperparameters = {
            'penalty': 'l2',
            'solver': 'lbfgs',
            'l1_ratio': None,
            'C': 1.0,
            'dual': False,
            'tol': 0.0001,
            'fit_intercept': True,
            'intercept_scaling': 1,
            'class_weight': None,
            'random_state': 10,
            'max_iter': 10**6,
            'verbose': 0,
            'warm_start': False,
            'n_jobs': -1
            }
        self.predictor = skLogReg(**self.hyperparameters)

    def to_dict(self):
        """Serialize the model into a dictionary."""

        # Get the parameter dictionary
        dictionary = deepcopy(self.inputs)

        # Loop over the object-related keys
        for key in (
                'dataset', 'preprocessor', 'tuner', 'inspector', 'evaluator'):

            # Check if a value is available
            if dictionary[key] is not None:

                # Serialize the attribute objects
                dictionary[key] = dictionary[key].to_dict()

        return {'Logistic Regression': dictionary}

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
            :class:`~pyanno4rt.learning.models.logistic._logistic_regression.LogisticRegression`
            The object used to represent the model.
        """

        # Deserialize the attribute objects
        dictionary['dataset'] = TabularDataset.from_dict(dictionary['dataset'])

        return cls(**dictionary)

    def load_data(self):
        """Load the dataset."""

        # Log a message about loading the dataset
        get_logger().info("Loading dataset for '%s' ...", self.label)

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
        get_logger().info(
            "Adding feature calculator for '%s' ...", self.label)

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
            proposal = self.tuner.search(deepcopy(self), features, labels)

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

        # Check if the proposal has a regularization subdictionary
        if 'regularization' in proposal:

            # Get the unpacked regularization parameters
            regularization = {**proposal['regularization']}

        else:

            # Get the regularization parameters directly
            regularization = {
                key: proposal[key] for key in (
                    'penalty', 'solver', 'l1_ratio', 'C')
                if key in proposal}

        # Build the hyperparameter dictionary
        self.hyperparameters = self.hyperparameters | {
            **regularization,
            'tol': proposal.get('tol', 0.0001),
            'class_weight': proposal.get('class_weight'),
            'n_jobs': -1 if regularization.get('solver') != 'liblinear' else 1}

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

        # Preprocess the features
        preprocessed_features, _ = self.preprocess(features)

        # Calculate the model prediction
        prediction = self.predict(preprocessed_features)

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
        predictor_gradient = (prediction-prediction**2)*self.predictor.coef_[0]

        return feature_gradient, preprocessing_gradient, predictor_gradient

    def load(self):
        """
        Load an external logistic regression model.

        Returns
        -------
        object of class :class:`~sklearn.linear_model.LogisticRegression`
            The object used to represent the prediction model.

        object of class :class:`~pyanno4rt.learning.preprocessing._tabular_preprocessor.TabularPreprocessor`
            The object used to represent the preprocessor.
        """

        # Log a message about the model file reading
        get_logger().info("Reading '%s' model from file ...", self.label)

        return (
            load(open(self.path+'/predictor.sav', 'rb')),
            load(open(self.path+'/preprocessor.sav', 'rb')))

    def save(
            self,
            path):
        """
        Save the model.

        Parameters
        ----------
        path : str
            Path for storing the logistic regression model.
        """

        # Open a file stream for the predictor
        with open(path+'/predictor.sav', 'wb') as file:

            # Dump the predictor
            dump(self.predictor, file)

        # Open a file stream for the preprocessor
        with open(path+'/preprocessor.sav', 'wb') as file:

            # Dump the preprocessor
            dump(self.preprocessor, file)

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
                partial(validate_type, options=(type(None), TabularDataset)),
                ),
            'preprocessor': (
                partial(validate_type, options=(
                    type(None), TabularPreprocessor)),
                ),
            'tuner': (
                partial(validate_type, options=(
                    type(None), *TUNERS.values())),
                ),
            'inspector': (
                partial(validate_type, options=(
                    type(None), ModelInspector)),
                ),
            'evaluator': (
                partial(validate_type, options=(
                    type(None), ModelEvaluator)),
                ),
            'path': (
                partial(validate_type, options={
                    True: str,
                    False: (type(None), str)},
                    condition=inputs['dataset'] is None),
                partial(validate_path)
                )
            }

        # Loop over the dictionary items
        for key, value in inputs.items():

            # Loop over the validation functions
            for function in validation_map[key]:

                # Run the validation function
                function(key, value)
