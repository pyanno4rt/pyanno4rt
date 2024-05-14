"""Model inspection."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from pyanno4rt.datahub import Datahub

# %% Internal package import

from pyanno4rt.learning_model.inspection.algorithms import (
    permutation_importances)

# %% Class definition


class ModelInspector():
    """
    Model inspection class.

    This class provides the computation method for a number of inspection \
    algorithms on a machine learning outcome model.

    Parameters
    ----------
    model_label : str
        Label for the machine learning outcome model.

    Attributes
    ----------
    model_label : str
        See 'Parameters'.
    """

    def __init__(
            self,
            model_label):

        # Log a message about the initialization of the model inspector
        Datahub().logger.display_info("Initializing model inspector ...")

        # Get the model label from the arguments
        self.model_label = model_label

    def compute(
            self,
            model_instance,
            hyperparameters,
            features,
            labels,
            preprocessing_steps,
            number_of_repeats,
            oof_folds):
        """
        Compute the inspection results.

        Parameters
        ----------
        model_instance : object
            The object representing the machine learning outcome model.

        hyperparameters : dict
            Dictionary with the machine learning outcome model hyperparameters.

        features : ndarray
            Values of the input features.

        labels : ndarray
            Values of the input labels.

        preprocessing_steps : list
            Sequence of labels associated with preprocessing algorithms for \
            the machine learning outcome model.

        number_of_repeats : int
            Number of feature permutations.

        oof_folds : ndarray
            Out-of-fold split numbers.
        """

        # Enter the permutation importances into the datahub
        Datahub().model_inspections[self.model_label] = {
            'permutation_importance': permutation_importances(
                model_instance, hyperparameters, features, labels,
                preprocessing_steps, number_of_repeats, oof_folds)}
