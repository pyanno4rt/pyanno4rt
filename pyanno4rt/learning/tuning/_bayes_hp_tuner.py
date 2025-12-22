"""Bayesian hyperparameter tuning."""

# Author: Tim Ortkamp

# %% External package import

from statistics import mean

from copy import deepcopy
from functools import partial
from hyperopt import fmin, space_eval, STATUS_FAIL, STATUS_OK, Trials, tpe
from numpy import unique, where
from warnings import filterwarnings
filterwarnings(action='ignore')

# %% Internal package import

import pyanno4rt.learning._maps as maps
from pyanno4rt.logging import get_logger
from pyanno4rt.tools import filter_dict
from pyanno4rt.validation import (
    validate_item, validate_item_in_set, validate_type)

# %% Class definition


class BayesHPTuner():
    """
    Bayesian hyperparameter tuning class.

    Parameters
    ----------
    space : object of class \
        :class:`~pyanno4rt.learning.tuning.spaces._tune_space_rf.TuneSpaceRF`\
        :class:`~pyanno4rt.learning.tuning.spaces._tune_space_lr.TuneSpaceLR`\
        :class:`~pyanno4rt.learning.tuning.spaces._tune_space_nb.TuneSpaceNB`\
        :class:`~pyanno4rt.learning.tuning.spaces._tune_space_knn.TuneSpaceKNN`\
        :class:`~pyanno4rt.learning.tuning.spaces._tune_space_nn.TuneSpaceNN`\
        :class:`~pyanno4rt.learning.tuning.spaces._tune_space_svm.TuneSpaceSVM`\
        :class:`~pyanno4rt.learning.tuning.spaces._tune_space_dt.TuneSpaceDT`
        The object used to represent the hyperparameter search space.

    evaluations : int, default=25
        Number of evaluation steps (trials).

    score : {'AUC', 'Brier score', 'Logloss'}, default='AUC'
        Scoring function for the hyperparameter set evaluation.

    Attributes
    ----------
    space : object of class \
        :class:`~pyanno4rt.learning.tuning.spaces._tune_space_rf.TuneSpaceRF`\
        :class:`~pyanno4rt.learning.tuning.spaces._tune_space_lr.TuneSpaceLR`\
        :class:`~pyanno4rt.learning.tuning.spaces._tune_space_nb.TuneSpaceNB`\
        :class:`~pyanno4rt.learning.tuning.spaces._tune_space_knn.TuneSpaceKNN`\
        :class:`~pyanno4rt.learning.tuning.spaces._tune_space_nn.TuneSpaceNN`\
        :class:`~pyanno4rt.learning.tuning.spaces._tune_space_svm.TuneSpaceSVM`\
        :class:`~pyanno4rt.learning.tuning.spaces._tune_space_dt.TuneSpaceDT`
        See 'Parameters'.

    evaluations : int
        See 'Parameters'.

    score : {'AUC', 'Brier score', 'Logloss'}
        See 'Parameters'.

    _step : int
        Step counter.
    """

    def __init__(
            self,
            space,
            evaluations=25,
            score='AUC'):

        # Get the input arguments
        self.inputs = filter_dict(vars(), remove_keys=('self',))

        # Check the input arguments
        self.validate(self.inputs)

        # Get the instance attributes
        self.space = space
        self.evaluations = evaluations
        self.score = score

        # Initialize the step counter
        self._step = None

    def to_dict(self):
        """Serialize the Bayesian hyperparameter tuner into a dictionary."""

        # Get the parameter dictionary
        dictionary = deepcopy(self.inputs)

        # Serialize the tune space
        dictionary['space'] = self.space.to_dict()

        return dictionary|{'name': 'Bayes'}

    @classmethod
    def from_dict(
            cls,
            dictionary):
        """
        Deserialize the Bayesian hyperparameter tuner from a dictionary.

        Parameters
        ----------
        dictionary : dict
            Dictionary with the tuner parameters.

        Returns
        -------
        object of class \
            :class:`~pyanno4rt.learning.tuning._bayes_hp_tuner.BayesHPTuner`
            The object used to represent the tuner.
        """

        # Deserialize the tune space
        dictionary['space'] = (
            maps.TUNE_SPACES[dictionary['space'].pop('name')].from_dict(
                dictionary['space']))

        return cls(**dictionary)

    def search(
            self,
            model,
            features,
            labels,
            folds):
        """
        Search the hyperparameter tune space.

        Parameters
        ----------
        model : object of class \
            :class:`~pyanno4rt.learning.models.logistic._logistic_regression.LogisticRegression`
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

        def log_trial(step, trials):
            """Log the result of a trial."""

            get_logger().info(
                "Tuning hyperparameters (%s/%s) - best loss: %s ...",
                step, self.evaluations,
                round(min(filter(None, trials.losses())), 4))

        def objective(proposal, trials, space):
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

            # Loop over the past trials
            for trial in trials:

                # Check if the trial has been accepted
                if trial['result']['status'] == STATUS_OK:

                    # Filter the trial values
                    values = {
                        key: value[0] for key, value
                        in trial['misc']['vals'].items() if value}

                    # Check if the proposed set equals the trial set
                    if proposal == space_eval(space, values):

                        # Log a message about the tuning status
                        log_trial(self._step, trials)

                        # Increment the step variable
                        self._step += 1

                        # Return an error status
                        return {'status': STATUS_FAIL}

            # Update the hyperparameter set
            model.get_bayes_hp(proposal)

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

            # Check if the first evaluation step has been passed
            if self._step > 0:

                # Log a message about the tuning status
                log_trial(self._step, trials)

            # Increment the step variable
            self._step += 1

            return {
                'loss': mean(repeat_scores),
                'params': model.hyperparameters,
                'status': STATUS_OK}

        # Log a message about the hyperparameter tuning
        get_logger().info(
            "Performing Bayesian hyperparameter search with %s-fold "
            "cross-validation and %s repeat(s) ...",
            len(unique(folds)), folds.shape[1])

        # Initialize the step variable
        self._step = 0

        # Get the hyperopt search space
        hp_space = self.space.to_hyperopt()

        # Get the score function
        scorer = maps.LOSSES[self.score]

        # Generate a trials object for the evaluation history
        bayes_trials = Trials()

        # Run the hyperparameter tuning algorithm
        hyperparameters = fmin(
            fn=partial(objective, trials=bayes_trials, space=hp_space),
            space=hp_space,
            algo=tpe.suggest,
            max_evals=self.evaluations,
            trials=bayes_trials,
            return_argmin=False,
            verbose=False,
            show_progressbar=False)

        # Log a message about the tuning completion
        get_logger().info(
            "Completed hyperparameter tuning (%s/%s) - best loss: %s ... ",
            self._step, self.evaluations,
            round(min(filter(None, bayes_trials.losses())), 4))

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
            'space': (
                partial(validate_type, options=(*maps.TUNE_SPACES.values(),)),
                ),
            'evaluations': (
                partial(validate_type, options=int),
                partial(validate_item, reference=0, sign='>')
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
