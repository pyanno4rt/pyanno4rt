"""Logistic regression model."""

# Author: Tim Ortkamp

# %% External package import

from pickle import dump, load
from sklearn.linear_model import LogisticRegression as skLogReg

# %% Internal package import

from pyanno4rt.learning.models import MachineLearningModel

# %% Class definition


class LogisticRegression(MachineLearningModel):
    """
    Logistic regression model class.

    This class implements methods to handle logistic regression models.

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

    predictor : object of class \
        :class:`~sklearn.linear_model.LogisticRegression`
        The object used to represent the prediction model.

    Notes
    -----
    See :class:`~pyanno4rt.learning.models._machine_learning_model.MachineLearningModel`\
    for details on the inherited attributes.
    """

    # Initialize the hyperparameters
    hyperparameters = {
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
        'n_jobs': -1}

    # Initialize the predictor
    predictor = skLogReg(**hyperparameters)

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

    def _get_bayes_hp(
            self,
            proposal):
        """
        Get the hyperparameters from a Bayesian search proposal.

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

    def _get_random_hp(
            self,
            proposal):
        """
        Get the hyperparameters from a random search proposal.

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

        return (prediction-prediction**2)*self.predictor.coef_[0]

    def _load_predictor(self):
        """Load the predictor."""

        # Open a file stream for the predictor
        with open(self.model_path+'/predictor.sav', 'rb') as file:

            # Load the predictor
            self.predictor = load(file)

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
