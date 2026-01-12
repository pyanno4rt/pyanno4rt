"""Grid search hyperparameter tuning."""

# Author: Tim Ortkamp

# %% External package import

from statistics import mean
from warnings import filterwarnings

from copy import deepcopy
from functools import partial
from math import inf
from numpy import unique, where

# %% Internal package import

import pyanno4rt.learning._maps as maps
from pyanno4rt.logging import get_logger
from pyanno4rt.tools import filter_dict
from pyanno4rt.validation import validate_item_in_set, validate_type

# %% Set package options

filterwarnings(action='ignore')

# %% Class definition


class GridHPTuner():
    """
    Grid search hyperparameter tuning class.

    This class implements methods to perform grid-based hyperparameter tuning.

    Parameters
    ----------
    grid : object of class \
        :class:`~pyanno4rt.learning.tuning.grids._tune_grid_rf.TuneGridRF`\
        :class:`~pyanno4rt.learning.tuning.grids._tune_grid_lr.TuneGridLR`\
        :class:`~pyanno4rt.learning.tuning.grids._tune_grid_nb.TuneGridNB`\
        :class:`~pyanno4rt.learning.tuning.grids._tune_grid_knn.TuneGridKNN`\
        :class:`~pyanno4rt.learning.tuning.grids._tune_grid_nn.TuneGridNN`\
        :class:`~pyanno4rt.learning.tuning.grids._tune_grid_svm.TuneGridSVM`\
        :class:`~pyanno4rt.learning.tuning.grids._tune_grid_dt.TuneGridDT`
        The object used to represent the hyperparameter search grid.

    score : {'AUC', 'BCE', 'Brier', 'Dice', 'Focal BCE', 'KLD', 'Hinge'}, \
        default='AUC'
        Scoring function for the hyperparameter set evaluation.

    Attributes
    ----------
    arguments : dict
        Dictionary with the model input arguments (for serialization).

    grid : object of class \
        :class:`~pyanno4rt.learning.tuning.grids._tune_grid_rf.TuneGridRF`\
        :class:`~pyanno4rt.learning.tuning.grids._tune_grid_lr.TuneGridLR`\
        :class:`~pyanno4rt.learning.tuning.grids._tune_grid_nb.TuneGridNB`\
        :class:`~pyanno4rt.learning.tuning.grids._tune_grid_knn.TuneGridKNN`\
        :class:`~pyanno4rt.learning.tuning.grids._tune_grid_nn.TuneGridNN`\
        :class:`~pyanno4rt.learning.tuning.grids._tune_grid_svm.TuneGridSVM`\
        :class:`~pyanno4rt.learning.tuning.grids._tune_grid_dt.TuneGridDT`
        See 'Parameters'.

    score : {'AUC', 'BCE', 'Brier', 'Dice', 'Focal BCE', 'KLD', 'Hinge'}
        See 'Parameters'.

    _step : int
        Step counter.

    _current_best_loss : float
        Current best value of the loss function.
    """

    def __init__(
            self,
            grid,
            score='AUC'):

        # Get the input arguments
        self.arguments = filter_dict(vars(), remove_keys=('self',))

        # Check the input arguments
        self.validate(self.arguments)

        # Get the instance attributes
        self.grid = grid
        self.score = score

        # Initialize the step counter
        self._step = None

        # Initialize the current best loss
        self._current_best_loss = None

    def to_dict(self):
        """Serialize the hyperparameter tuner into a dictionary."""

        # Get the parameter dictionary
        dictionary = deepcopy(self.arguments)

        # Serialize the tune grid
        dictionary['grid'] = self.grid.to_dict()

        return dictionary|{'name': 'Grid'}

    @classmethod
    def from_dict(
            cls,
            dictionary):
        """
        Deserialize the hyperparameter tuner from a dictionary.

        Parameters
        ----------
        dictionary : dict
            Dictionary with the tuner parameters.

        Returns
        -------
        object of class \
            :class:`~pyanno4rt.learning.tuning._grid_hp_tuner.GridHPTuner`
            The object used to represent the tuner.
        """

        # Deserialize the tune grid
        dictionary['grid'] = (
            maps.TUNE_GRIDS[dictionary['grid'].pop('name')].from_dict(
                dictionary['grid']))

        return cls(**dictionary)

    def search(
            self,
            model,
            features,
            labels,
            folds):
        """
        Search the hyperparameter tune grid.

        Parameters
        ----------
        model : object of class \
            :class:`~pyanno4rt.learning._models.forest._random_forest.RandomForest`\
            :class:`~pyanno4rt.learning._models.logistic._logistic_regression.LogisticRegression`\
            :class:`~pyanno4rt.learning._models.naive_bayes._naive_bayes.NaiveBayes`\
            :class:`~pyanno4rt.learning._models.neighbors._k_nearest_neighbors.KNearestNeighbors`\
            :class:`~pyanno4rt.learning._models.neural_network._feed_forward_net.FeedForwardNet`\
            :class:`~pyanno4rt.learning._models.svm._support_vector_machine.SupportVectorMachine`\
            :class:`~pyanno4rt.learning._models.tree._decision_tree.DecisionTree`
            The object used to represent the tunable model.

        features : ndarray
            Values of the input features.

        labels : ndarray
            Values of the input labels.

        folds : ndarray
            Fold numbers for cross-validation.

        Returns
        -------
        dict
            Dictionary with the values of the tuned hyperparameters.
        """

        def log_trial():
            """Log the result of a trial."""

            get_logger().info(
                "Tuning hyperparameters (%s/%s) - best loss: %s ...",
                self._step, number_of_evaluations,
                round(self._current_best_loss, 4))

        def objective(proposal):
            """Compute the objective function for a hyperparameter set."""

            def compute_fold_score(indices):
                """Compute the score for a single cross-validation split."""

                # Get the training and validation split
                split = [
                    features[indices[0]], labels[indices[0]],
                    features[indices[1]], labels[indices[1]]]

                # Fit the preprocessor on the training split
                model.fit_preprocessor(split[0], split[1])

                # Transform both splits
                split = [
                    *model.preprocess(split[0], split[1]),
                    *model.preprocess(split[2], split[3])]

                # Fit the predictor on the training split
                model.fit_predictor(split[0], split[1])

                # Compute the training and validation scores
                scores = [
                    scorer(labels, model.predict(features))
                    for features, labels in (split[:2], split[2:])]

                return max(scores)

            # Update the hyperparameter set
            model.update_hyperparameters(proposal)

            # Compute the objective function value (score) across all folds
            repeat_scores = (mean(map(compute_fold_score, (
                ((training_indices, validation_indices)
                 for training_indices, validation_indices in (
                         (where(folds[:, index] != number),
                          where(folds[:, index] == number))
                         for number in set(folds[:, index])))
                if len(set(folds[:, index])) > 2
                else ((where(folds[:, index] != 1),
                       where(folds[:, index] == 1)),))))
                for index in range(folds.shape[1]))

            # Get the loss
            loss = mean(repeat_scores)

            # Update the current best loss
            self._current_best_loss = min(self._current_best_loss, loss)

            # Check if the first evaluation step has been passed
            if self._step > 0:

                # Log a message about the tuning status
                log_trial()

            # Increment the step variable
            self._step += 1

            return {
                'loss': loss,
                'params': model.hyperparameters}

        # Log a message about the hyperparameter tuning
        get_logger().info(
            "Performing grid-based hyperparameter search with %s-fold "
            "cross-validation and %s repeat(s) ...",
            len(unique(folds)), folds.shape[1])

        # Initialize the step variable
        self._step = 0

        # Initialize the current best loss
        self._current_best_loss = inf

        # Get the search grid
        grid = self.grid.to_grid()

        # Get the number of evaluations
        number_of_evaluations = len(grid)

        # Get the score function
        scorer = maps.LOSSES[self.score]

        # Get the best loss and hyperparameters
        loss, hyperparameters = min(
            (objective(proposal) for proposal in grid),
            key=lambda x: x['loss']).values()

        # Log a message about the tuning completion
        get_logger().info(
            "Completed hyperparameter tuning (%s/%s) - best loss: %s ... ",
            self._step, number_of_evaluations, round(loss, 4))

        return hyperparameters

    def validate(
            self,
            inputs):
        """
        Validate the input arguments.

        Parameters
        ----------
        inputs : dict
            Dictionary with the input arguments.
        """

        # Get the validation map
        validation_map = {
            'grid': (
                partial(validate_type, options=(*maps.TUNE_GRIDS.values(),)),
                ),
            'score': (
                partial(validate_type, options=str),
                partial(validate_item_in_set, options=(*maps.LOSSES,))
                )
            }

        # Loop over the inputs
        for key, value in inputs.items():

            # Loop over the validation functions
            for function in validation_map[key]:

                # Run the validation function
                function(key, value)
