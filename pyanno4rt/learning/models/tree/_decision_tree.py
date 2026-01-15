"""Decision tree model."""

# Author: Tim Ortkamp

# %% External package import

from pickle import dump, load
from warnings import filterwarnings

from numpy import zeros
from sklearn.tree import DecisionTreeClassifier

# %% Internal package import

from pyanno4rt.learning.models import MachineLearningModel
from pyanno4rt.learning.models.tree import ProjectionTree
from pyanno4rt.validation import validate_item_in_set, validate_type

# %% Set package options

filterwarnings(action='ignore')

# %% Class definition


class DecisionTree(MachineLearningModel):
    """
    Decision tree model class.

    This class implements methods to handle decision tree models.

    Parameters
    ----------
    label : str
        Label for the learning model.

    diff_mode : {'project', 'soften'}
        The strategy used to approximate gradients at discrete boundaries.

        Currently available:

            - 'project': approximates the gradient by projecting onto the \
                nearest lower-value manifold.

            - 'soften': approximates the gradient by learning a soft \
                representation with sigmoidal transitions.

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
    diff_mode : {'project', 'soften'}
        See 'Parameters'.

    hyperparameters : dict
        Dictionary with the model hyperparameters.

    predictor : object of class :class:`~sklearn.tree.DecisionTreeClassifier`
        The object used to represent the prediction model.

    surrogate : object of class \
        :class:`~pyanno4rt.learning.models.tree._projection_tree.ProjectionTree`\
        :class:`~pyanno4rt.learning.models.tree._soft_tree.SoftTree`
        The object used to represent the differentiable surrogate model.

    Notes
    -----
    See :class:`~pyanno4rt.learning.models._machine_learning_model.MachineLearningModel`\
    for details on the inherited attributes.
    """

    def __init__(
            self,
            label,
            diff_mode,
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

        # Get the differentiation strategy
        self.diff_mode = diff_mode

        # Validate the differentiation strategy
        validate_type('diff_mode', diff_mode, str)
        validate_item_in_set('diff_mode', diff_mode, ('project', 'soften'))

        # Extend the input arguments
        self.arguments |= {'diff_mode': diff_mode}

        # Initialize the hyperparameters
        self.hyperparameters = {
            'criterion': 'gini',
            'splitter': 'best',
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'min_weight_fraction_leaf': 0.0,
            'max_features': None,
            'random_state': 13,
            'max_leaf_nodes': None,
            'min_impurity_decrease': 0.0,
            'class_weight': None,
            'ccp_alpha': 0.0,
            'monotonic_cst': None}

        # Initialize the predictor
        self.predictor = DecisionTreeClassifier(**self.hyperparameters)

        # Check if gradients should be approximated by projecting
        if diff_mode == 'project':

            # Initialize the projection tree
            self.surrogate = ProjectionTree()

        # Check if gradients should be approximated by softening
        # elif diff_mode == 'soften':

            # Initialize the soft tree
            # self.surrogate = SoftTree()

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

        # Check if the tree has been parsed to the surrogate
        if self.surrogate.is_parsed:

            # Return the approximate gradient
            return self.surrogate.gradientize(preprocessed_features)

        # Return a zero gradient
        return zeros(preprocessed_features.shape[1])

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
