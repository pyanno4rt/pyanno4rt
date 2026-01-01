"""Random forest model."""

# Author: Tim Ortkamp

# %% External package import

from pickle import dump, load

from numpy import ones
from sklearn.ensemble import RandomForestClassifier

# %% Internal package import

from pyanno4rt.learning.models import MachineLearningModel

# %% Class definition


class RandomForest(MachineLearningModel):
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
    hyperparameters : dict
        Dictionary with the model hyperparameters.

    predictor : object of class \
        :class:`~sklearn.ensemble.RandomForestClassifier`
        The object used to represent the prediction model.

    Notes
    -----
    See :class:`~pyanno4rt.learning.models._machine_learning_model.MachineLearningModel`\
    for details on the inherited attributes.
    """

    # Initialize the hyperparameters
    hyperparameters = {
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

    # Initialize the predictor
    predictor = RandomForestClassifier(**hyperparameters)

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

        return ones((len(preprocessed_features),))

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
