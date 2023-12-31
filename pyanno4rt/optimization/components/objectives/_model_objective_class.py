"""Model objective template."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from abc import ABCMeta, abstractmethod
from math import inf

# %% Internal package import

from pyanno4rt.datahub import Datahub

# %% Class definition


class ModelObjectiveClass(metaclass=ABCMeta):
    """Model objective template class."""

    def __init__(
            self,
            name,
            parameter_name,
            parameter_category,
            model_parameters,
            embedding,
            weight,
            link,
            identifier,
            display):

        # Get the class arguments
        class_arguments = locals()

        # Remove the 'self'-key from the class arguments
        class_arguments.pop('self')

        # Check the class attributes and objective parameters
        Datahub().input_checker.approve(class_arguments)

        # Get the fixed objective attributes
        self.name = name
        self.parameter_name = parameter_name
        self.parameter_category = parameter_category

        # Get the model parameters
        self.model_parameters = {
            'model_label': model_parameters.get('model_label'),
            'model_folder_path': model_parameters.get('model_folder_path'),
            'data_path': model_parameters.get('data_path'),
            'feature_filter': model_parameters.get(
                'feature_filter', [[], 'remove']),
            'label_viewpoint': model_parameters.get(
                'label_viewpoint', 'long-term'),
            'label_bounds': model_parameters.get(
                'label_bounds', [1.5, inf]),
            'fuzzy_matching': model_parameters.get('fuzzy_matching', True),
            'preprocessing_steps': model_parameters.get(
                'preprocessing_steps', ('Equalizer',)),
            'architecture': model_parameters.get(
                'architecture', 'input-convex'),
            'max_hidden_layers': model_parameters.get('max_hidden_layers', 4),
            'tune_space': model_parameters.get('tune_space', {}),
            'tune_evaluations': model_parameters.get(
                'tune_evaluations', 250),
            'tune_score': model_parameters.get('tune_score', 'log_loss'),
            'tune_splits': model_parameters.get('tune_splits', 5),
            'inspect_model': model_parameters.get('inspect_model', False),
            'evaluate_model': model_parameters.get(
                'evaluate_model', False),
            'oof_splits': model_parameters.get('oof_splits', 5),
            'write_features': model_parameters.get(
                'write_features', True),
            'write_gradients': model_parameters.get(
                'write_gradients', False),
            'display_options': model_parameters.get(
                'display_options', {'graphs': ['AUC-ROC', 'AUC-PR', 'F1'],
                                    'kpis': ['Logloss', 'Brier score',
                                             'Subset accuracy', 'Cohen Kappa',
                                             'Hamming loss', 'Jaccard score',
                                             'Precision', 'Recall', 'F1 score',
                                             'MCC', 'AUC']})
            }

        # Check the model parameters
        Datahub().input_checker.approve(self.model_parameters)

        # Get the variable objective parameters
        self.embedding = embedding
        self.weight = float(weight)
        self.link = [] if link is None else link
        self.identifier = identifier
        self.display = display

        # Initialize the adjustment indicator
        self.adjusted_parameters = False

        # Indicate the model dependency of the objective
        self.DEPENDS_ON_MODEL = True

    def get_parameter_value(self):
        """
        Get the value of the parameters.

        Returns
        -------
        tuple
            Value of the parameters.
        """
        return self.parameter_value

    def set_parameter_value(
            self,
            *args):
        """
        Set the value of the parameters.

        Parameters
        ----------
        args : tuple
            Keyworded parameters. args[0] should give the value to be set.
        """
        self.parameter_value = args[0]

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
           *args):
        """
        Set the value of the weight.

        Parameters
        ----------
        args : tuple
            Keyworded parameters. args[0] should give the value to be set.
        """
        self.weight = args[0]

    @abstractmethod
    def add_model(self):
        """Add the learning model to the objective function."""

    @abstractmethod
    def compute_objective_value(
            self,
            *args):
        """Compute the value of the objective function."""

    @abstractmethod
    def compute_gradient_value(
            self,
            *args):
        """Compute the value of the gradient."""
