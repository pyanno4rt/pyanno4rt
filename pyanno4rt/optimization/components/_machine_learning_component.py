"""Machine learning component template."""

# Author: Tim Ortkamp

# %% External package import

from abc import ABCMeta, abstractmethod

# %% Internal package import

from pyanno4rt.datahub import Datahub
from pyanno4rt.tools import compare_dictionaries

# %% Class definition


class MachineLearningComponent(metaclass=ABCMeta):
    """
    Machine learning component template class.

    Parameters
    ----------
    name : str
        Name of the component class.

    segment : str
        Name of the segment associated with the component.

    parameter_name : tuple
        Name of the component parameters.

    parameter_category : tuple
        Category of the component parameters.

    model_parameters : dict
        Dictionary with the data handling & learning model parameters:

        - model_label : str
            Label for the machine learning model.

        - model_folder_path : None or str, default=None
            Path to a folder for loading an external machine learning model.

        - data_path : str
            Path to the data set used for fitting the machine learning model.

        - data_columns : dict
            Dictionary with the column information on features and label.

        - preprocessing_steps : list, default=['Identity']
            Sequence of labels associated with preprocessing algorithms to \
            preprocess the input features.

            The following preprocessing steps are currently available:

            - 'Identity' \
                :class:`~pyanno4rt.learning_model.preprocessing.transformers._identity.Identity`
            - 'StandardScaler' \
                :class:`~pyanno4rt.learning_model.preprocessing.transformers._standard_scaler.StandardScaler`
            - 'Whitening' \
                :class:`~pyanno4rt.learning_model.preprocessing.transformers._whitening.Whitening`

        - architecture : {'vanilla-input-convex', 'vanilla'}, default='vanilla'
            Type of architecture for the neural network model.

        - max_hidden_layers : int, default=2
            Maximum number of hidden layers for the neural network model.

        - tune_space : dict, default={}
            Search space for the Bayesian hyperparameter optimization.

        - tune_evaluations : int, default=50
            Number of evaluation steps (trials) for the Bayesian \
            hyperparameter optimization.

        - tune_score : {'AUC', 'Brier score', 'Logloss'}, default='Logloss'
            Scoring function for the evaluation of the hyperparameter set \
            candidates.

        - tune_splits : int, default=5
            Number of splits for the stratified cross-validation within each \
            hyperparameter optimization step.

        - tune_repeats : int, default=1
            Number of repeats for the stratified cross-validation within each \
            hyperparameter optimization step.

        - inspect_model : bool, default=False
            Indicator for the inspection of the machine learning model.

        - evaluate_model : bool, default=False
            Indicator for the evaluation of the machine learning model.

        - oof_splits : int, default=5
            Number of splits for the stratified cross-validation within the \
            out-of-folds evaluation step.

        - oof_repeats : int, default=1
            Number of repeats for the stratified cross-validation within the \
            out-of-folds evaluation step.

        - write_features : bool, default=False
            Indicator for writing the iteratively calculated feature vectors \
            into a feature history.

        - display_options : dict, \
            default={'graphs': ['AUC-ROC', 'AUC-PR', 'F1'],\
                     'kpis': ['Logloss', 'Brier score', 'Subset accuracy', \
                              'Cohen Kappa', 'Hamming loss', 'Jaccard score', \
                              'Precision', 'Recall', 'F1 score', 'MCC', 'AUC']}
            Dictionary with the graph and KPI display options.

    embedding : {'active', 'passive'}
        Mode of embedding for the component. In 'passive' mode, the component \
        value is computed and tracked, but not considered in the optimization \
        problem, unlike in 'active' mode.

    weight : int or float
        Weight of the component function.

    rank : int, default=1
        Rank of the component in the lexicographic order.

    bounds : None or list
        Constraint bounds for the component.

    link : None or list
        Other segments used for joint evaluation.

    identifier : None or str
        Additional string for naming the component.

    display : bool
        Indicator for the display of the component.

    Attributes
    ----------
    name : str
        See 'Parameters'.

    segment : str
        See 'Parameters'.

    parameter_name : tuple
        See 'Parameters'.

    parameter_category : tuple
        See 'Parameters'.

    parameter_value : list
        Value of the component parameters.

    embedding : {'active', 'passive'}
        See 'Parameters'.

    weight : float
        See 'Parameters'.

    rank : int
        See 'Parameters'.

    bounds : list
        See 'Parameters'.

    link : None or list
        See 'Parameters'.

    identifier : None or str
        See 'Parameters'.

    display : bool
        See 'Parameters'.

    model_parameters : dict
        See 'Parameters'.

    data_model_handler : None
        Initial variable for the object used to handle the dataset, the \
        feature map generation and the feature (re-)calculation.

    model : None
        Initial variable for the object used to preprocess, tune, train, \
        inspect and evaluate the machine learning model.

    adjusted_parameters : bool
        Indicator for the adjustment of the parameters due to fractionation.
    """

    def __init__(
            self,
            name,
            segment,
            parameter_name,
            parameter_category,
            model_parameters,
            embedding,
            weight,
            rank,
            bounds,
            link,
            identifier,
            display):

        # Get the class arguments
        class_arguments = locals()

        # Remove the 'self'-key from the class arguments dictionary
        class_arguments.pop('self')

        # Initialize the datahub
        hub = Datahub()

        # Check the class attributes
        hub.input_checker.approve(class_arguments)

        # Set the instance attributes from the class arguments
        self.name = name
        self.segment = segment
        self.parameter_name = parameter_name
        self.parameter_category = parameter_category
        self.parameter_value = []
        self.embedding = embedding
        self.weight = float(weight)
        self.rank = rank
        self.bounds = (
            (0.0, 1.0) if bounds is None or embedding == 'passive' else (
                0.0 if bounds[0] is None else float(bounds[0]),
                1.0 if bounds[1] is None else float(bounds[1])))
        self.link = [] if link is None else link
        self.identifier = identifier
        self.display = display

        # Set the model parameters
        self.model_parameters = {
            'model_label': model_parameters.get('model_label'),
            'model_folder_path': model_parameters.get('model_folder_path'),
            'data_path': model_parameters.get('data_path'),
            'data_columns': model_parameters.get('data_columns'),
            'preprocessing_steps': model_parameters.get(
                'preprocessing_steps', ['Identity']),
            'architecture': model_parameters.get('architecture', 'vanilla'),
            'max_hidden_layers': model_parameters.get('max_hidden_layers', 2),
            'tune_space': model_parameters.get('tune_space', {}),
            'tune_evaluations': model_parameters.get('tune_evaluations', 50),
            'tune_score': model_parameters.get('tune_score', 'Logloss'),
            'tune_splits': model_parameters.get('tune_splits', 5),
            'tune_repeats': model_parameters.get('tune_repeats', 1),
            'inspect_model': model_parameters.get('inspect_model', False),
            'evaluate_model': model_parameters.get('evaluate_model', False),
            'oof_splits': model_parameters.get('oof_splits', 5),
            'oof_repeats': model_parameters.get('oof_repeats', 1),
            'write_features': model_parameters.get('write_features', False),
            'display_options': model_parameters.get(
                'display_options', {
                    'graphs': ['AUC-ROC', 'AUC-PR', 'F1'],
                    'kpis': [
                        'Logloss', 'Brier score', 'Subset accuracy',
                        'Cohen Kappa', 'Hamming loss', 'Jaccard score',
                        'Precision', 'Recall', 'F1 score', 'MCC', 'AUC']})
            }

        # Check the model parameters
        hub.input_checker.approve(self.model_parameters)

        # Check the tune space
        hub.input_checker.approve(self.model_parameters['tune_space'])

        # Check the model display options
        hub.input_checker.approve(self.model_parameters['display_options'])

        # Initialize the data model handler and the outcome model
        self.data_model_handler = None
        self.model = None

        # Initialize the adjustment indicator
        self.adjusted_parameters = False

    def __eq__(
            self,
            other):
        """
        Compare an instance with another object.

        Parameters
        ----------
        other : object
            The object to compare the instance with.

        Returns
        -------
        bool
            Indicator for the equality of the objects.
        """

        return (
            all(self.__dict__[key] == other.__dict__[key] for key in (
                'name', 'link', 'identifier'))
            and compare_dictionaries(
                self.__dict__.get('model_parameters', {}),
                other.__dict__.get('model_parameters', {})))

    def get_class(self):
        """
        Get the name of the component class.

        Returns
        -------
        str
            Name of the component class.
        """

        return 'MachineLearningComponent'

    def get_parameter_value(self):
        """
        Get the value of the parameters.

        Returns
        -------
        list
            Value of the parameters.
        """

        return self.parameter_value

    def set_parameter_value(
            self,
            value):
        """
        Set the value of the parameters.

        Parameters
        ----------
        value : list
            Value to be set.
        """

        self.parameter_value = value

    def get_weight_value(self):
        """
        Get the value of the weight.

        Returns
        -------
        float
            Value of the weight.
        """

        return self.weight

    def set_weight_value(
           self,
           value):
        """
        Set the value of the weight.

        Parameters
        ----------
        value : float
            Value to be set.
        """

        self.weight = value

    @abstractmethod
    def add_model(self):
        """Add the machine learning model to the component."""

    @abstractmethod
    def compute_value(
            self,
            dose,
            segment):
        """Compute the component value."""

    @abstractmethod
    def compute_gradient(
            self,
            dose,
            segment):
        """Compute the component gradient."""
