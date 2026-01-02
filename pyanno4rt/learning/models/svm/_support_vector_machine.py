"""Support vector machine model."""

# Author: Tim Ortkamp

# %% External package import

from pickle import dump, load
from warnings import filterwarnings

from sklearn.svm import SVC

# %% Internal package import

from pyanno4rt.learning.models import MachineLearningModel
from pyanno4rt.learning.models.svm import (
    linear_gradient, poly_gradient, rbf_gradient, sigmoid_gradient)

# %% Set package options

filterwarnings(action='ignore')

# %% Class definition


class SupportVectorMachine(MachineLearningModel):
    """
    Support vector machine model class.

    This class implements methods to handle support vector machine models.

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
    hyperparameters : dict
        Dictionary with the model hyperparameters.

    predictor : object of class :class:`~sklearn.svm.SVC`
        The object used to represent the prediction model.

    multiplier : float
        Multiplicative parameter of the Platt scaling function.

    summand : float
        Additive parameter of the Platt scaling function.

    gradient : None or callable
        Gradient function for the fitted kernel type.

    Notes
    -----
    See :class:`~pyanno4rt.learning.models._machine_learning_model.MachineLearningModel`\
    for details on the inherited attributes.
    """

    # Initialize the hyperparameters
    hyperparameters = {
        'C': 1.0,
        'kernel': 'rbf',
        'degree': 3,
        'gamma': 'scale',
        'coef0': 0.0,
        'shrinking': True,
        'probability': True,
        'tol': 0.001,
        'cache_size': 200,
        'class_weight': None,
        'verbose': False,
        'max_iter': -1,
        'decision_function_shape': 'ovr',
        'break_ties': False,
        'random_state': 11}

    # Initialize the predictor
    predictor = SVC(**hyperparameters)

    def __init__(
            self,
            label,
            dataset,
            preprocessor=None,
            tuner=None,
            inspector=None,
            evaluator=None,
            model_path=None):

        # Call the superclass constructor
        super().__init__(
            label=label,
            dataset=dataset,
            preprocessor=preprocessor,
            tuner=tuner,
            inspector=inspector,
            evaluator=evaluator,
            model_path=model_path)

        # Initialize the prediction model attributes
        self.multiplier, self.summand = None, None
        self.gradient = None

    def _get_space_hp(
            self,
            proposal):
        """
        Get the hyperparameters from a search space proposal.

        Parameters
        ----------
        proposal : dict
            Proposal for the tunable hyperparameters.
        """

        # Build the hyperparameter dictionary
        self.hyperparameters = self.hyperparameters | {
            'C': proposal.get('C', 1.0),
            'kernel': proposal.get('kernel', 'rbf'),
            'degree': proposal.get('degree', 3),
            'gamma': proposal.get('gamma', 'scale'),
            'tol': proposal.get('tol', 0.001),
            'class_weight': proposal.get('class_weight')}

    def _get_grid_hp(
            self,
            proposal):
        """
        Get the hyperparameters from a grid search proposal.

        Parameters
        ----------
        proposal : dict
            Proposal for the tunable hyperparameters.
        """

        self._get_space_hp(proposal)

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

        # Get the Platt scaling parameters
        self.multiplier, self.summand = (
            -self.predictor.probA_[0], -self.predictor.probB_[0])

        # Map the kernel types to the gradient functions
        gradient_map = {
            'linear': linear_gradient, 'poly': poly_gradient,
            'rbf': rbf_gradient, 'sigmoid': sigmoid_gradient}

        # Get the gradient function
        self.gradient = gradient_map[self.predictor.kernel]

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

    def _predictor_gradient(
            self,
            preprocessed_features):
        """
        Calculate the predictor gradient.

        Parameters
        ----------
        preprocessed_features : ndarray
            Values of the preprocessed input features.

        Returns
        -------
        ndarray
            Predictor gradient w.r.t the preprocessed features.
        """

        # Calculate the model prediction
        prediction = self.predict(preprocessed_features)

        return (
            self.multiplier*prediction*(1-prediction)
            * self.gradient(self.predictor, preprocessed_features))

    def _load_predictor(self):
        """Load the predictor."""

        # Open a file stream for the predictor
        with open(self.model_path+'/predictor.sav', 'rb') as file:

            # Load the predictor
            self.predictor = load(file)

        # Get the Platt scaling parameters
        self.multiplier, self.summand = (
            -self.predictor.probA_[0], -self.predictor.probB_[0])

        # Map the kernel types to the gradient functions
        gradient_map = {
            'linear': linear_gradient, 'poly': poly_gradient,
            'rbf': rbf_gradient, 'sigmoid': sigmoid_gradient}

        # Get the gradient function
        self.gradient = gradient_map[self.predictor.kernel]

    def _load_hyperparameters(self):
        """Load the hyperparameters from the predictor."""

        # Get the hyperparameters
        self.hyperparameters = self.predictor.get_params()

    def _save_predictor(
            self,
            path):
        """
        Save the predictor.

        Parameters
        ----------
        path : str
            Path for storing the predictor.
        """

        # Open a file stream for the predictor
        with open(path+'/predictor.sav', 'wb') as file:

            # Dump the predictor
            dump(self.predictor, file)
