"""Soft decision tree model."""

# Author: Tim Ortkamp

# %% External package import

from pickle import dump, load
from warnings import filterwarnings

# %% Internal package import

from pyanno4rt.learning.models import MachineLearningModel
from pyanno4rt.learning.models.tree import SoftTree

# %% Set package options

filterwarnings(action='ignore')

# %% Class definition


class SoftDecisionTree(MachineLearningModel):
    """
    Soft decision tree model class.

    This class implements methods to handle soft decision tree models.

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
        :class:`~pyanno4rt.learning.tuning._randomized_hp_tuner.RandomizedHPTuner`,\
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
        :class:`~pyanno4rt.learning.models.tree._soft_tree.SoftTree`
        The object used to represent the prediction model.

    Notes
    -----
    See :class:`~pyanno4rt.learning.models._machine_learning_model.MachineLearningModel`\
    for details on the inherited attributes.
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

        # Call the superclass constructor
        super().__init__(
            label=label,
            dataset=dataset,
            preprocessor=preprocessor,
            tuner=tuner,
            inspector=inspector,
            evaluator=evaluator,
            model_path=model_path)

        # Initialize the hyperparameters
        self.hyperparameters = {
            'criterion': 'log_loss',
            'max_depth': None,
            'temperature': [0.01, 2.0],
            'tolerance': 1e-3}

        # Initialize the predictor
        self.predictor = SoftTree(**self.hyperparameters)

    def update_hyperparameters(
            self,
            proposal):
        """
        Update the hyperparameters from a search proposal.

        Parameters
        ----------
        proposal : dict
            Proposal for the tunable hyperparameters.
        """

        # Update the hyperparameters
        self.hyperparameters |= {
            key: proposal[key]
            for key in proposal.keys() & self.hyperparameters.keys()}

    def fit_predictor(
            self,
            features,
            labels):
        """
        Fit the model.

        Parameters
        ----------
        features : ndarray
            Feature values.

        labels : ndarray
            Label values.
        """

        # Initialize the predictor
        self.predictor = self.predictor.set_params(**self.hyperparameters)

        # Fit the predictor
        self.predictor.fit(features, labels)

    def predict(
            self,
            features):
        """
        Predict the label value(s).

        Parameters
        ----------
        features : ndarray
            Feature values.

        Returns
        -------
        float or ndarray
            Predicted label value(s).
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
            Preprocessed feature values.

        Returns
        -------
        ndarray
            Predictor gradient w.r.t the preprocessed features.
        """

        # Calculate the gradient
        gradient = self.predictor.gradientize(preprocessed_features)

        # Check if the return should be 1D
        if gradient.shape[0] == 1:

            # Return the 1D gradient
            return gradient.reshape(-1)

        # Return the 2D gradient
        return gradient

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
