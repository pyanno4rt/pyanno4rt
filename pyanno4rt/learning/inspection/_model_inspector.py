"""Model inspection."""

# Author: Tim Ortkamp

# %% External package import

from functools import partial

# %% Internal package import

import pyanno4rt.learning._maps as maps
from pyanno4rt.learning.inspection import permutation_importances
from pyanno4rt.tools import filter_dict
from pyanno4rt.validation import (
    validate_item, validate_item_in_set, validate_type)

# %% Class definition


class ModelInspector():
    """
    Model inspection class.

    Parameters
    ----------
    pi_score : {'AUC', 'Brier score', 'Logloss'}, default='AUC'
        Permutation importance score.

    pi_permutations : int, default=20
        Number of permutations for permutation importance.

    pi_splits : int, default=5
        Number of splits for multi-run permutation importance.

    pi_repeats : int, default=1
        Number of repeats for multi-run permutation importance.

    Attributes
    ----------
    pi_score : {'AUC', 'Brier score', 'Logloss'}
        See 'Parameters'.

    pi_permutations : int
        See 'Parameters'.

    pi_splits : int
        See 'Parameters'.

    pi_repeats : int
        See 'Parameters'.

    results : dict
        Dictionary with the model inspection results.
    """

    def __init__(
            self,
            pi_score='AUC',
            pi_permutations=20,
            pi_splits=5,
            pi_repeats=1):

        # Get the input arguments
        self.inputs = filter_dict(vars(), remove_keys=('self',))

        # Check the input arguments
        self.validate(self.inputs)

        # Get the input attributes
        self.pi_score = pi_score
        self.pi_permutations = pi_permutations
        self.pi_splits = pi_splits
        self.pi_repeats = pi_repeats

        # Initialize the result dictionary
        self.results = {}

    def to_dict(self):
        """Serialize the model inspector into a dictionary."""

    @classmethod
    def from_dict(
            cls,
            dictionary):
        """Deserialize the model inspector from a dictionary."""

    def run(
            self,
            model):
        """
        Inspect a model.

        Parameters
        ----------
        model : object of class \
            :class:`~pyanno4rt.learning._models.forest._random_forest.RandomForest`\
            :class:`~pyanno4rt.learning._models.logistic._logistic_regression.LogisticRegression`\
            :class:`~pyanno4rt.learning._models.naive_bayes._naive_bayes.NaiveBayes`\
            :class:`~pyanno4rt.learning._models.neighbors._k_nearest_neighbors.KNearestNeighbors`\
            :class:`~pyanno4rt.learning._models.network._neural_network.NeuralNetwork`\
            :class:`~pyanno4rt.learning._models.svm._support_vector_machine.SupportVectorMachine`\
            :class:`~pyanno4rt.learning._models.tree._decision_tree.DecisionTree`
            The object used to represent the outcome model.
        """

        # Compute the permutation importances
        self.results['permutation_importances'] = permutation_importances(
            model, self.pi_score, self.pi_permutations, self.pi_splits,
            self.pi_repeats)

    def validate(
            self,
            inputs):
        """
        Validate the input arguments.

        Parameters
        ----------
        inputs : dict
            Dictionary with the mappings between argument names and values.
        """

        validation_map = {
            'pi_score': (
                partial(validate_type, options=str),
                partial(validate_item_in_set, options=(*maps.LOSSES,))
                ),
            'pi_permutations': (
                partial(validate_type, options=int),
                partial(validate_item, reference=1, sign='>=')
                ),
            'pi_splits': (
                partial(validate_type, options=int),
                partial(validate_item, reference=1, sign='>=')
                ),
            'pi_repeats': (
                partial(validate_type, options=int),
                partial(validate_item, reference=1, sign='>=')
                )
            }

        # Loop over the dictionary items
        for key, value in inputs.items():

            # Loop over the validation functions
            for function in validation_map[key]:

                # Run the validation function
                function(key, value)
