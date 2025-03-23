"""Naive Bayes TCP component."""

# Author: Tim Ortkamp

# %% External package import

from copy import deepcopy
from numpy import exp, log, pi, size
from numpy import sum as nsum
from scipy.special import logsumexp

# %% Internal package import

from pyanno4rt.datahub import Datahub
from pyanno4rt.learning import DataModelHandler
from pyanno4rt.learning.naive_bayes import NaiveBayesModel
from pyanno4rt.optimization.components import MachineLearningComponent
from pyanno4rt.tools import filter_dict

# %% Class definition


class NaiveBayesTCP(MachineLearningComponent):
    """
    Naive Bayes TCP component class.

    This class provides methods to compute the value and the gradient of the \
    naive Bayes TCP component, as well as to add the naive Bayes model.

    Parameters
    ----------
    segment : str
        Name of the segment associated with the component.

    model_parameters : dict
        Dictionary with the data handling & learning model parameters, see \
        the class
        :class:`~pyanno4rt.optimization.components._machine_learning_component_class.MachineLearningComponentClass`.

    component_type : {'constraint', 'objective'}, default='objective'
        Type of the component.

    embedding : {'active', 'passive'}, default='active'
        Mode of embedding for the component. In 'passive' mode, the component \
        value is computed and tracked, but not considered in the optimization \
        problem, unlike in 'active' mode.

    weight : int or float, default=1.0
        Weight of the component function.

    rank : int, default=1
        Rank of the component in the lexicographic order.

    bounds : None or list, default=None
        Constraint bounds for the component.

    link : None or list, default=None
        Other segments used for joint evaluation.

    identifier : None or str, default=None
        Additional string for naming the component.

    display : bool, default=True
        Indicator for the display of the component.

    Attributes
    ----------
    arguments : dict
        Dictionary with the component input arguments (for serialization).

    data_model_handler : object of class \
        :class:`~pyanno4rt.learning_model._data_model_handler.DataModelHandler`
        The object used to handle the dataset, the feature map generation and \
        the feature (re-)calculation.

    model : object of class \
        :class:`~pyanno4rt.learning_model.frequentist._naive_bayes.NaiveBayesModel`
        The object used to preprocess, tune, train, inspect and evaluate the \
        naive Bayes model.

    parameter_value : list
        Naive Bayes model parameters.

    bounds : list
        See 'Parameters'.
    """

    def __init__(
            self,
            segment,
            model_parameters,
            component_type='objective',
            embedding='active',
            weight=1.0,
            rank=1,
            bounds=None,
            link=None,
            identifier=None,
            display=True):

        # Call the superclass constructor to initialize and check attributes
        super().__init__(
            name='Naive Bayes TCP',
            segment=segment,
            component_type=component_type,
            parameter_name=(),
            parameter_category=(),
            model_parameters=model_parameters,
            embedding=embedding,
            weight=weight,
            rank=rank,
            bounds=bounds,
            link=link,
            identifier=identifier,
            display=display)

        # Set the input arguments
        self.arguments = filter_dict(
            locals(), remove_keys=('self', '__class__'))

    def to_dict(self):
        """Return the component input dictionary."""

        # Get the parameter dictionary
        dictionary = deepcopy(self.arguments)

        # Serialize the data columns
        dictionary['model_parameters']['data_columns'] = [
            item.to_dict()
            for item in dictionary['model_parameters']['data_columns']]

        return {'Naive Bayes TCP': dictionary}

    def add_model(self):
        """Add the naive Bayes model to the component."""

        # Initialize the datahub
        hub = Datahub()

        # Log a message about the model addition
        hub.logger.display_info(
            f"Adding naive Bayes model for '{self.name}' ...")

        # Initialize the data model handler
        self.data_model_handler = DataModelHandler(
            model_label=self.model_parameters['model_label'],
            model_folder_path=self.model_parameters['model_folder_path'],
            data_path=self.model_parameters['data_path'],
            data_columns=self.model_parameters['data_columns'],
            tune_splits=self.model_parameters['tune_splits'],
            tune_repeats=self.model_parameters['tune_repeats'],
            oof_splits=self.model_parameters['oof_splits'],
            oof_repeats=self.model_parameters['oof_repeats'],
            write_features=self.model_parameters['write_features'])

        # Integrate the model-related classes
        self.data_model_handler.integrate()

        # Initialize the naive Bayes model
        self.model = NaiveBayesModel(
            model_label=self.model_parameters['model_label'],
            model_folder_path=self.model_parameters['model_folder_path'],
            dataset=hub.datasets[self.model_parameters['model_label']],
            preprocessing_steps=self.model_parameters['preprocessing_steps'],
            tune_space=self.model_parameters['tune_space'],
            tune_evaluations=self.model_parameters['tune_evaluations'],
            tune_score=self.model_parameters['tune_score'],
            inspect_model=self.model_parameters['inspect_model'],
            evaluate_model=self.model_parameters['evaluate_model'],
            display_options=self.model_parameters['display_options'])

        # Get the naive Bayes model parameters
        self.parameter_value = []

        # Transform the component bounds
        self.bounds = sorted(-bound for bound in self.bounds)

    def compute_value(
            self,
            dose,
            segment):
        """
        Compute the function value.

        Parameters
        ----------
        dose : tuple
            Tuple with the dose values.

        segment : tuple
            Tuple with the segment names.

        Returns
        -------
        float
            Function value.
        """

        # Compute the feature vector
        raw_features = self.data_model_handler.feature_calculator.featurize(
            dose, segment)

        # Preprocess the feature vector
        preprocessed_features = self.model.preprocess(raw_features)

        return -self.model.predict(
            preprocessed_features, self.model.prediction_model)

    def compute_gradient(
            self,
            dose,
            segment):
        """
        Compute the gradient vector.

        Parameters
        ----------
        dose : tuple
            Tuple with the dose values.

        segment : tuple
            Tuple with the segment names.

        Returns
        -------
        ndarray
            Gradient vector.
        """

        def calculate_model_gradient(features):
            """Calculate the naive Bayes model gradient."""

            # Get the number of classes
            number_of_classes = size(self.model.prediction_model.classes_)

            # Get the fitted mean and variance parameters
            means = self.model.prediction_model.theta_
            variances = self.model.prediction_model.var_

            # Calculate the joint log likelihood value for all classes
            joint_log_likelihood = [
                log(self.model.prediction_model.class_prior_[i])
                - 0.5*nsum(log(2*pi*variances[i, :]))
                - 0.5*nsum(
                    ((preprocessed_features - means[i, :])**2)
                    / (variances[i, :]), 1)
                for i in range(number_of_classes)]

            # Calculate the joint log likelihood gradient for all classes
            joint_log_likelihood_gradient = [
                (-1*(features-means[i, :]) / variances[i, :])
                for i in range(number_of_classes)]

            # Calculate the log evidence gradient
            log_evidence_gradient = (
                nsum(
                    joint_log_likelihood_gradient[i]
                    * exp(joint_log_likelihood[i])
                    for i in range(number_of_classes))
                / nsum(
                    exp(joint_log_likelihood[i])
                    for i in range(number_of_classes)))

            # Calculate the probability prediction from the model
            prediction = exp(
                joint_log_likelihood[1][0] - logsumexp(joint_log_likelihood))

            # Calculate the input feature gradient
            gradient = prediction * (
                joint_log_likelihood_gradient[1] - log_evidence_gradient)

            return gradient.reshape(-1)

        # Get the feature calculator
        feature_calculator = self.data_model_handler.feature_calculator

        # Compute the feature vector
        raw_features = feature_calculator.featurize(dose, segment)

        # Preprocess the feature vector
        preprocessed_features = self.model.preprocess(raw_features)

        # Compute the model gradient
        model_gradient = -calculate_model_gradient(preprocessed_features)

        # Compute the preprocessing pipeline gradient
        preprocessing_gradient = (
            self.model.preprocessor.gradientize(raw_features))

        # Compute the feature gradient
        feature_gradient = feature_calculator.gradientize(dose, segment)

        return (model_gradient * preprocessing_gradient) @ feature_gradient
