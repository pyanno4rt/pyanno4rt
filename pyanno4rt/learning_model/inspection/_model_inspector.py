"""Model inspection."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from pyanno4rt.datahub import Datahub

# %% Internal package import

from pyanno4rt.learning_model.inspection.inspections import (
    PermutationImportance)

# %% Class definition


class ModelInspector():
    """
    Model inspection class.

    This class provides a collection of inspection methods to be computed in \
    a single method call.

    Parameters
    ----------
    model_name : string
        Name of the learning model.

    hyperparameters : dict
        Hyperparameters dictionary.

    Attributes
    ----------
    model_name : string
        See 'Parameters'.

    hyperparameters : dict, default = None
        See 'Parameters'.

    inspections : dict
        Dictionary with the inspection values.
    """

    def __init__(
            self,
            model_name,
            model_class,
            hyperparameters=None):

        # Log a message about the initialization of the model inspector
        Datahub().logger.display_info("nitializing model inspector ...")

        # Get the instance attributes from the arguments
        self.model_name = model_name
        self.model_class = model_class
        self.hyperparameters = hyperparameters

        # Initialize the dictionary with the inspection results
        self.inspections = {}

    def compute(
            self,
            model,
            features,
            labels,
            number_of_repeats):
        """
        Compute the inspection results.

        Parameters
        ----------
        model : object
            Instance of the outcome prediction model.

        features : ndarray
            Values of the input features.

        labels : ndarray
            Values of the input labels.

        number_of_repeats : int
            Number of feature permutations to evaluate.
        """
        # Add the training permutation importances
        self.inspections[
            'permutation_importance_training'] = (
                PermutationImportance(
                    self.model_name, self.model_class,
                    self.hyperparameters).compute(model, features, labels,
                                                  number_of_repeats))

        # Add the validation permutation importances
        self.inspections['permutation_importance_validation'] = (
            PermutationImportance(
                self.model_name, self.model_class,
                self.hyperparameters).compute_oof(model, features, labels,
                                                  number_of_repeats))
