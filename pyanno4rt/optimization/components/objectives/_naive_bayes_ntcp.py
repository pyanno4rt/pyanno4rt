"""Naive Bayes NTCP objective."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from numpy import array, zeros

# %% Internal package import

from pyanno4rt.datahub import Datahub
from pyanno4rt.learning_model import DataModelHandler
from pyanno4rt.learning_model.frequentist import NaiveBayesModel
from pyanno4rt.optimization.components.objectives import ModelObjectiveClass

# %% Class definition


class NaiveBayesNTCP(ModelObjectiveClass):
    """
    Naive Bayes NTCP objective class.

    This class provides methods to compute the value and the gradient of the \
    naive Bayes NTCP objective, as well as to add the naive Bayes model and \
    to get/set the parameters and the objective weight.

    Parameters
    ----------
    model_parameters : dict
        Dictionary with the data handling & learning model parameters:

        - ``data_path`` : string, path to the data set used for fitting the \
            naive Bayes model;
        - ``feature_filter`` : tuple, default = ((), 'remove'), (sub)set \
            of the feature names as a tuple and a value from `{'retain', \
            'remove'} as an indicator for retaining or removing the (sub)set \
            prior to fitting the naive Bayes model;
        - ``label_viewpoint`` : {'early', 'late', 'long-term', \
            'longitudinal', 'profile'}, default = 'long-term', time of \
            observation for the presence of tumor control and/or normal \
            tissue complication events;
        - ``label_bounds`` : list, default = [1.5, inf], bounds for the \
            label values to binarize them into positive and negative class \
            (values within the specified bounds are classified as positive);
        - ``fuzzy_matching`` : bool, default = True, indicator for the use of \
            fuzzy string matching (if False, exact string matching is \
            applied) to generate the mapping between features, segmented \
            structures and calculation functions;
        - ``preprocessing_steps`` : tuple, default = ('Equalizer',), sequence \
            of labels associated with preprocessing algorithms which make up \
            the preprocessing pipeline for the naive Bayes model;
        - ``architecture`` : {'input-convex', 'standard'}, default = \
            'input-convex', type of architecture for the neural network \
            model;
        - ``max_hidden_layers`` : int, default = 4, maximum number of hidden \
            layers for the neural network model;
        - ``tune_space`` : dict, default = {}, search space for the Bayesian \
            hyperparameter optimization;
        - ``tune_evaluations`` : int, default = 250, number of evaluation \
            steps (trials) for the Bayesian hyperparameter optimization;
        - ``tune_score`` : string, default = 'log_loss', scoring function \
            for the evaluation of the hyperparameter set candidates;
        - ``tune_splits`` : int, default = 5, number of splits for the \
            stratified cross-validation within each hyperparameter \
            optimization step;
        - ``inspect_model`` : bool, default = False, indicator for the \
            inspection of the naive Bayes model;
        - ``evaluate_model`` : bool, default = False, indicator for the \
            evaluation of the naive Bayes model;
        - ``oof_splits`` : int, default = 5, number of splits for the \
            stratified cross-validation within the out-of-folds evaluation \
            step of the naive Bayes model;
        - ``write_features`` : bool, default = False, indicator for writing \
            the iteratively computed feature vectors to the feature history;
        - ``write_gradients`` : bool, default = False, indicator for writing \
            the iteratively computed feature gradient matrices to the \
            gradient history, should be set to True only for analysis \
            purposes due to the potentially high memory requirements.

        .. note::
           The gradient matrices are stored in sparse mode, but might still
           have high memory consumption, depending on the number of features \
           in the dataset and the number of voxels in the dose grid. It is \
           recommended to set ``write_grad`` = True only for analysis purposes.

           More information on some of the parameters, e.g. the available \
           preprocessing algorithms, can be found in the model classes. \
           Especially, if no hyperparameter search space is defined, the \
           related outcome model will implicitly apply default values, which \
           can be read from the class file.

    embedding : {'active', 'passive'}, default = 'active'
        Mode of embedding for the objective. In 'passive' mode, the objective \
        value is computed and tracked, but not included in the optimization \
        problem. In 'active' mode, however, both objective value and \
        gradient vector are computed and included in the optimization problem.

    weight : int or float, default = 1.0
        Weight of the objective function.

    link : list, default = None
        Link to additional segments for joint evaluation.

    Attributes
    ----------
    name : string
        Name of the objective class.

    parameter_name : tuple
        Name of the objective parameters.

    parameter_category : tuple
        Category of the objective parameters.

    model_parameters : dict
        See 'Parameters'.

    embedding : {'active', 'passive'}
        See 'Parameters'.

    weight : float
        See 'Parameters'.

    link : list
        See 'Parameters'.

    adjusted_parameters : bool
        Indicator for the adjustment of the dose-related parameters.

    DEPENDS_ON_MODEL : bool
        Indicator for the model dependency of the objective.

    data_model_handler : object of class `DataModelHandler`
        Instance of the class `DataModelHandler`, which provides methods for \
        handling the data set, the learning model, and their integration \
        back into the optimization problem.

    model : object of class `NaiveBayesModel`
        Instance of the class `NaiveBayesModel`, which holds methods to \
        preprocess, tune, train, inspect and evaluate the naive Bayes model.

    parameter_value : list
        Value of the naive Bayes model parameters.
    """

    def __init__(
            self,
            model_parameters,
            embedding='active',
            weight=1.0,
            link=None,
            identifier=None,
            display=True):

        # Call the superclass constructor to initialize and check attributes
        super().__init__(name='Naive Bayes NTCP',
                         parameter_name=(),
                         parameter_category=(),
                         model_parameters=model_parameters,
                         embedding=embedding,
                         weight=weight,
                         link=link,
                         identifier=identifier,
                         display=display)

        # Set the individual parameter value
        self.parameter_value = []

    def add_model(self):
        """Add the naive Bayes model to the objective."""
        # Initialize the datahub
        hub = Datahub()

        # Log a message about the model addition
        hub.logger.display_info("Adding naive Bayes model for '{}' ..."
                                .format(self.name))

        # Initialize the data model handler
        self.data_model_handler = DataModelHandler(
            data_path=self.model_parameters['data_path'],
            feature_filter=self.model_parameters['feature_filter'],
            label_viewpoint=self.model_parameters['label_viewpoint'],
            label_bounds=self.model_parameters['label_bounds'],
            fuzzy_matching=self.model_parameters['fuzzy_matching'],
            write_features=self.model_parameters['write_features'],
            write_gradients=self.model_parameters['write_gradients'])

        # Get the naive Bayes model
        self.model = NaiveBayesModel(
            model_label=self.model_parameters['model_label'],
            model_folder_path=self.model_parameters['model_folder_path'],
            dataset=self.data_model_handler.data.dataset,
            preprocessing_steps=self.model_parameters['preprocessing_steps'],
            tune_space=self.model_parameters['tune_space'],
            tune_evaluations=self.model_parameters['tune_evaluations'],
            tune_score=self.model_parameters['tune_score'],
            tune_splits=self.model_parameters['tune_splits'],
            inspect_model=self.model_parameters['inspect_model'],
            evaluate_model=self.model_parameters['evaluate_model'],
            oof_splits=self.model_parameters['oof_splits'])

        # Add the dataset to the datahub
        hub.datasets[self.model_parameters['model_label']] = (
            self.data_model_handler.data.dataset)

        # Add the feature map to the datahub
        hub.feature_maps[self.model_parameters['model_label']] = (
            self.data_model_handler.feature_map_generator.feature_map)

    def compute_objective_value(
            self,
            *args):
        """
        Compute the value of the objective.

        Parameters
        ----------
        args : tuple
            Keyworded parameters. args[0] should give the dose vector(s) to \
            evaluate, args[1] the corresponding segment(s).

        Returns
        -------
        float
            Value of the objective function.
        """
        # Featurize the dose and segment inputs
        raw_features = self.data_model_handler.feature_calculator.featurize(
            args[0], args[1])

        # Run the specific preprocessing pipeline of the model
        preprocessed_features = self.model.preprocess(raw_features)

        return self.model.predict(preprocessed_features)

    def compute_gradient_value(
            self,
            *args):
        """
        Compute the value of the gradient.

        Parameters
        ----------
        args : tuple
            Keyworded parameters. args[0] should give the dose vector(s) to \
            evaluate, args[1] the corresponding segment(s).

        Returns
        -------
        ndarray
            Gradient vector of the objective function.

        Notes
        -----
        The objective gradient is described by end-to-end chain ruling from \
        the model input gradient across the preprocessing pipeline gradient \
        to the feature gradient.
        """
        # Get the feature calculator
        feature_calculator = self.data_model_handler.feature_calculator

        # Featurize the dose and segment inputs
        raw_features = feature_calculator.featurize(args[0], args[1])

        # Get the model gradient from the coefficient vector
        model_gradient = zeros((raw_features.shape[0],)) 

        # Get the preprocessing gradient from the pipeline
        preprocessing_gradient = array(
            self.model.preprocessor.gradientize(raw_features))

        # Get the feature gradient from the feature definitions
        feature_gradient = feature_calculator.gradientize(args[0], args[1])

        return (model_gradient * preprocessing_gradient) @ feature_gradient
