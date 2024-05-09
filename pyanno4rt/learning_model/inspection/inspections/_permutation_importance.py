"""Permutation importance."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from concurrent.futures import ThreadPoolExecutor
from numpy import vstack
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from tensorflow.compat.v2.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau)
from tensorflow.compat.v2.keras.models import clone_model

# %% Internal package import

from pyanno4rt.datahub import Datahub
from pyanno4rt.learning_model.frequentist.additional_files import (
    loss_map, optimizer_map)

# %% Class definition


class PermutationImportance():
    """
    Permutation importance class.

    Parameters
    ----------
    model_name : string
        Name of the learning model.

    hyperparameters : dict, default = None
        Hyperparameters dictionary.

    Attributes
    ----------
    model_name : string
        See 'Parameters'.

    hyperparameters : dict
        See 'Parameters'.
    """

    def __init__(
            self,
            model_name,
            model_class,
            hyperparameters=None):

        # Get the instance attributes from the arguments
        self.model_name = model_name
        self.model_class = model_class
        self.hyperparameters = hyperparameters

    def compute(
            self,
            model,
            features,
            labels,
            number_of_repeats):
        """
        Compute the training permutation importance.

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

        Returns
        -------
        ndarray
            Permutation importance values per repetition.
        """
        # Log a message about the permutation importance computation
        Datahub().logger.display_info("Computing training permutation "
                                      "importance ...")

        # Compute the permutation importance
        importance = permutation_importance(estimator=model, X=features,
                                            y=labels, scoring=self.score,
                                            n_repeats=number_of_repeats,
                                            random_state=42)

        return importance['importances'].T

    def compute_oof(
            self,
            model,
            features,
            labels,
            number_of_repeats):
        """
        Compute the validation permutation importance.

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

        Returns
        -------
        tuple
            Permutation importance values per repetition and fold.
        """
        # Log a message about the permutation importance computation
        Datahub().logger.display_info("Computing validation permutation "
                                      "importance with 5-fold "
                                      "cross-validation ...")

        def compute_fold_importances(args):
            """Compute the out-of-folds labels for a single fold."""
            # Get the function parameters from the arguments
            training_indices = args[0]
            validation_indices = args[1]

            # Get the training and validation splits
            train_validation_split = (features[training_indices],
                                      features[validation_indices],
                                      labels[training_indices],
                                      labels[validation_indices])

            # Check if the model is either LR or SVM
            if self.model_class in ('Decision Tree',
                                    'K-Nearest Neighbors',
                                    'Logistic Regression',
                                    'Naive Bayes',
                                    'Random Forest',
                                    'Support Vector Machine'):

                # Fit the model with the training split
                model_clone = clone(model)
                model_clone.fit(train_validation_split[0],
                                train_validation_split[2])

            # Else, check if the model is NN
            elif self.model_class == 'Neural Network':

                # Fit the model with the training split
                model_clone = clone_model(model)

                # Compile the model
                model_clone.compile(
                    optimizer=optimizer_map[self.hyperparameters[
                        'optimizer']](
                        self.hyperparameters['learning_rate']),
                    loss=loss_map[self.hyperparameters['loss']]())

                # Fit the model with the training split
                model_clone.fit(train_validation_split[0],
                                train_validation_split[2],
                                batch_size=self.hyperparameters['batch_size'],
                                epochs=self.hyperparameters['epochs'],
                                validation_data=(train_validation_split[1],
                                                 train_validation_split[3]),
                                verbose=0,
                                callbacks=[
                                    ReduceLROnPlateau(
                                        monitor='val_loss',
                                        min_delta=0,
                                        factor=self.hyperparameters[
                                            'ReduceLROnPlateau_factor'],
                                        patience=self.hyperparameters[
                                            'ReduceLROnPlateau_patience'],
                                        mode='min',
                                        verbose=0),
                                    EarlyStopping(
                                        monitor='val_loss',
                                        min_delta=0,
                                        patience=self.hyperparameters[
                                            'EarlyStopping_patience'],
                                        mode='min',
                                        baseline=None,
                                        restore_best_weights=True,
                                        verbose=0)
                                    ],
                                class_weight={
                                    0: (1/sum(train_validation_split[2] == 0)
                                        * (2*len(train_validation_split[2]))),
                                    1: (1/sum(train_validation_split[2] == 1)
                                        * (2*len(train_validation_split[2])))})

            # Compute the permutation importance for the validation split
            importance = permutation_importance(estimator=model_clone,
                                                X=train_validation_split[1],
                                                y=train_validation_split[3],
                                                scoring=self.score,
                                                n_repeats=number_of_repeats,
                                                n_jobs=1, random_state=42)

            return importance['importances'].T

        # Initialize the stratified k-fold cross-validator
        cross_validator = StratifiedKFold(n_splits=5, random_state=4,
                                          shuffle=True)

        # Compute the permutation importances across all folds
        importances_per_fold = tuple(ThreadPoolExecutor().map(
            compute_fold_importances,
            ((train_indices, validation_indices)
             for (train_indices, validation_indices)
             in cross_validator.split(features, labels))))

        return vstack(importances_per_fold)

    def score(self,
              model,
              features,
              true_labels):
        """
        Create a callable for scoring the model.

        Parameters
        ----------
        model : object
            Instance of the outcome prediction model.

        features : ndarray
            Values of the input features.

        labels : ndarray
            Values of the input labels.

        Returns
        -------
        float
            Score of the loss function.
        """
        # Check if the model either LR or SVM
        if self.model_class in ('Decision Tree',
                                'K-Nearest Neighbors',
                                'Logistic Regression',
                                'Naive Bayes',
                                'Random Forest',
                                'Support Vector Machine'):

            # Check if the feature array has only a single row
            if features.shape[0] == 1:

                # Predict a single label prediction value
                predicted_labels = model.predict_proba(features)[0][1]

            else:

                # Predict an array with label predictions
                predicted_labels = model.predict_proba(features)[:, 1]

        # Else, check if the model is NN
        elif self.model_class == 'Neural Network':

            # Check if the feature array has only a single row
            if features.shape[0] == 1:

                # Predict a single label prediction value
                predicted_labels = model.predict(features, verbose=0)[0][0]

            else:

                # Predict an array with label predictions
                predicted_labels = model.predict(features, verbose=0)[:, 0]

        return roc_auc_score(true_labels, predicted_labels)
