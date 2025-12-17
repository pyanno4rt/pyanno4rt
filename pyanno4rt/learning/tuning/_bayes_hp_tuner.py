"""Bayesian hyperparameter tuning."""

# Author: Tim Ortkamp

# %% External package import

from statistics import mean

from copy import deepcopy
from functools import partial
from hyperopt import fmin, space_eval, STATUS_FAIL, STATUS_OK, Trials, tpe
from numpy import where, zeros
from sklearn.model_selection import RepeatedStratifiedKFold

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


    Attributes
    ----------

    """

    def __init__(
            self,
            space,
            evaluations=25,
            score='AUC',
            splits=5,
            repeats=1):

        # Get the input arguments
        self.inputs = filter_dict(vars(), remove_keys=('self',))

        # Check the input arguments
        self.validate(self.inputs)

        # Get the instance attributes
        self.space = space
        self.evaluations = evaluations
        self.score = score
        self.splits = splits
        self.repeats = repeats

        # Initialize the step counter for the hyperparameter search
        self._step = None

    def to_dict(self):
        """Serialize the tuner into a dictionary."""

        # Get the parameter dictionary
        dictionary = deepcopy(self.inputs)

        # Serialize the space
        dictionary['space'] = self.space.to_dict()

        return {'Bayes': dictionary}

    @classmethod
    def from_dict(
            cls,
            dictionary):
        """
        Deserialize the tuner from a dictionary.

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
        dictionary['space'] = maps.TUNE_SPACES['Logistic Regression'](
            **dictionary['space'])

        return cls(**dictionary)

    def search(
            self,
            model,
            features,
            labels):
        """
        Tune the hyperparameters of the machine learning model via sequential \
        model-based optimization using tree-structured Parzen estimators and \
        robust evaluation using stratified k-fold cross-validation.

        Parameters
        ----------
        model : object of class \
            :class:`~pyanno4rt.learning.models.logistic._logistic_regression.LogisticRegression`
            The object used to represent the machine learning model.

        features : ndarray
            Values of the input features.

        labels : ndarray
            Values of the input labels.

        Returns
        -------
        dict
            Dictionary with the values of the tuned hyperparameters.
        """

        # Define the output string function
        def log_trial(step, trials):
            """Log the result of a single trial."""

            get_logger().info(
                "Tuning hyperparameters (%s/%s) - best loss: %s ...",
                step, self.evaluations,
                round(min(filter(None, trials.losses())), 4))

        def objective(proposal, trials, space):
            """Compute the objective function for a set of hyperparameters."""

            def compute_fold_score(indices):
                """Compute the score for a single train-validation split."""

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
            "cross-validation and %s repeat(s) ...", self.splits, self.repeats)

        # Initialize the step variable
        self._step = 0

        # Get the hyperopt search space
        hp_space = self.space.to_hyperopt()

        # Map the score labels to the score functions
        scorer = maps.LOSSES[self.score]

        # Get the tune folds
        folds = self.get_folds(features, labels)

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

        # Log a message about the tuning status
        get_logger().info(
            "Completed hyperparameter tuning (%s/%s) - best loss: %s ... ",
            self._step, self.evaluations,
            round(min(filter(None, bayes_trials.losses())), 4))

        return hyperparameters

    def get_folds(
            self,
            features,
            labels):
        """
        Get the fold numbers for cross-validation.

        Parameters
        ----------
        features : ndarray
            Values of the input features.

        labels : ndarray
            Values of the input labels.

        Returns
        -------
        ndarray
            Fold numbers.
        """

        # Clamp the number of splits
        clamped_n_splits = min(self.splits, sum(labels))

        # Initialize the stratified k-fold cross-validator
        cross_validator = RepeatedStratifiedKFold(
            n_splits=5 if clamped_n_splits == 1 else clamped_n_splits,
            n_repeats=self.repeats, random_state=3)

        # Get the stratification splits
        splits = tuple(cross_validator.split(features, labels))

        # Divide the splits into chunks (for each repeat)
        chunks = [
            splits[index:index+self.splits] for index in range(
                0, clamped_n_splits*self.repeats, clamped_n_splits)]

        # Initialize the fold numbers
        folds = zeros((len(labels), self.repeats))

        # Loop over the chunks
        for column, chunk in enumerate(chunks):

            # Loop over the chunk splits
            for number, (_, validation_index) in enumerate(chunk):

                # Enter the fold number for the validation set repetition
                folds[validation_index, column] = (
                    int(number) if self.splits != 1 else 1)

        return folds

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
                ),
            'splits': (
                partial(validate_type, options=int),
                partial(validate_item, reference=1, sign='>=')
                ),
            'repeats': (
                partial(validate_type, options=int),
                partial(validate_item, reference=1, sign='>=')
                )
            }

        # Loop over the inputs
        for key, value in inputs.items():

            # Loop over the validation functions
            for function in validation_map[key]:

                # Run the validation function
                function(key, value)
