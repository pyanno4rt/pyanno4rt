"""Support vector machine TCP objective."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from numpy import array

# %% Internal package import

from pyanno4rt.datahub import Datahub
from pyanno4rt.learning_model import DataModelHandler
from pyanno4rt.learning_model.frequentist import SupportVectorMachineModel
from pyanno4rt.learning_model.frequentist.additional_files import (
    linear_decision_function, rbf_decision_function, poly_decision_function,
    sigmoid_decision_function, linear_decision_gradient, rbf_decision_gradient,
    poly_decision_gradient, sigmoid_decision_gradient)
from pyanno4rt.optimization.components.objectives import ModelObjectiveClass

# %% Class definition


class SupportVectorMachineTCP(ModelObjectiveClass):
    """
    Support vector machine TCP objective class.

    This class provides methods to compute the value and the gradient of the \
    support vector machine TCP objective, as well as to add the support \
    vector machine model and to get/set the parameters and the objective \
    weight.

    Parameters
    ----------
    model_parameters : dict
        Dictionary with the data handling & learning model parameters:

        - ``data_path`` : string, path to the data set used for fitting the \
            support vector machine model;
        - ``feature_filter`` : tuple, default = ((), 'remove'), (sub)set \
            of the feature names as a tuple and a value from `{'retain', \
            'remove'} as an indicator for retaining or removing the (sub)set \
            prior to fitting the support vector machine model;
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
            the preprocessing pipeline for the support vector machine model;
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
            inspection of the support vector machine model;
        - ``evaluate_model`` : bool, default = False, indicator for the \
            evaluation of the support vector machine model;
        - ``oof_splits`` : int, default = 5, number of splits for the \
            stratified cross-validation within the out-of-folds evaluation \
            step of the support vector machine model;
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

    decision_map : dict
        Mapping between the kernel types and the respective decision \
        functions and gradients.

    data_model_handler : object of class `DataModelHandler`
        Instance of the class `DataModelHandler`, which provides methods for \
        handling the data set, the learning model, and their integration \
        back into the optimization problem.

    model : object of class `SupportVectorMachineModel`
        Instance of the class `SupportVectorMachineModel`, which holds \
        methods to preprocess, tune, train, inspect and evaluate the support \
        vector machine model.

    parameter_value : list
        Value of the support vector machine parameters.
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
        super().__init__(name='Support Vector Machine TCP',
                         parameter_name=('w/alpha',),
                         parameter_category=('coefficient',),
                         model_parameters=model_parameters,
                         embedding=embedding,
                         weight=weight,
                         link=link,
                         identifier=identifier,
                         display=display)

        # Map the fitted kernel type to the decision function
        self.decision_map = {
            'linear': (linear_decision_function, linear_decision_gradient),
            'rbf': (rbf_decision_function, rbf_decision_gradient),
            'poly': (poly_decision_function, poly_decision_gradient),
            'sigmoid': (sigmoid_decision_function, sigmoid_decision_gradient)}

    def add_model(self):
        """Add the support vector machine model to the objective."""
        # Initialize the datahub
        hub = Datahub()

        # Log a message about the model addition
        hub.logger.display_info("Adding support vector machine model for '{}' "
                                "..."
                                .format(self.name))

        # Initialize the data model handler
        self.data_model_handler = DataModelHandler(
            model_label=self.model_parameters['model_label'],
            data_path=self.model_parameters['data_path'],
            feature_filter=self.model_parameters['feature_filter'],
            label_viewpoint=self.model_parameters['label_viewpoint'],
            label_bounds=self.model_parameters['label_bounds'],
            fuzzy_matching=self.model_parameters['fuzzy_matching'],
            write_features=self.model_parameters['write_features'],
            write_gradients=self.model_parameters['write_gradients'])

        # Get the support vector machine model
        self.model = SupportVectorMachineModel(
            model_label=self.model_parameters['model_label'],
            model_folder_path=self.model_parameters['model_folder_path'],
            dataset=hub.datasets[self.model_parameters['model_label']],
            preprocessing_steps=self.model_parameters['preprocessing_steps'],
            tune_space=self.model_parameters['tune_space'],
            tune_evaluations=self.model_parameters['tune_evaluations'],
            tune_score=self.model_parameters['tune_score'],
            tune_splits=self.model_parameters['tune_splits'],
            inspect_model=self.model_parameters['inspect_model'],
            evaluate_model=self.model_parameters['evaluate_model'],
            oof_splits=self.model_parameters['oof_splits'],
            display_options=self.model_parameters['display_options'])

        # Check if the linear kernel is used
        if self.model.prediction_model.kernel == 'linear':

            # Get the primal coefficients
            self.parameter_value = self.model.prediction_model.coef_[
                0].tolist()

        else:

            # Get the dual coefficients
            self.parameter_value = self.model.prediction_model.dual_coef_[
                0].tolist()

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

        Notes
        -----
        The objective function is described by the decision function of the \
        support vector machine model (without Platt scaling, convexity \
        depends on the kernel type).
        """
        # Featurize the dose and segment inputs
        raw_features = self.data_model_handler.feature_calculator.featurize(
            args[0], args[1])

        # Run the specific preprocessing pipeline of the model
        preprocessed_features = self.model.preprocess(raw_features)

        # Get the decision function for the SVM
        decision_function = self.decision_map[
            self.model.prediction_model.kernel][0]

        return -(decision_function(self.model.prediction_model,
                                   preprocessed_features))

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

        # Run the specific preprocessing pipeline of the model
        preprocessed_features = self.model.preprocess(raw_features)

        # Get the gradient function for the SVM
        gradient_function = self.decision_map[
            self.model.prediction_model.kernel][1]

        # Get the model gradient from the gradient function
        model_gradient = -gradient_function(self.model.prediction_model,
                                            preprocessed_features).reshape(-1)

        # Get the preprocessing gradient from the pipeline
        preprocessing_gradient = array(
            self.model.preprocessor.gradientize(raw_features))

        # Get the feature gradient from the feature definitions
        feature_gradient = feature_calculator.gradientize(args[0], args[1])

        return (model_gradient * preprocessing_gradient) @ feature_gradient
