"""Permutation importance computation."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from numpy import vstack, where
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import clone_model

# %% Internal package import

from pyanno4rt.datahub import Datahub
from pyanno4rt.learning_model.frequentist.addons import loss_map, optimizer_map
from pyanno4rt.learning_model.preprocessing import DataPreprocessor

# %% Function definition


def permutation_importances(model_instance, hyperparameters, features, labels,
                            preprocessing_steps, number_of_repeats, oof_folds):
    """
    Compute the permutation importances.

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
        Sequence of labels associated with the preprocessing algorithms for \
        the machine learning outcome model.

    number_of_repeats : int
        Number of feature permutations.

    oof_folds : ndarray
        Out-of-fold split numbers.

    Returns
    -------
    dict
        Dictionary with the training and out-of-folds permutation importances.
    """

    def compute_fold_importances(indices):
        """Compute the out-of-folds importances for a single fold."""

        # Get the training and validation splits
        train_validate_split = [features[indices[0]], features[indices[1]],
                                labels[indices[0]], labels[indices[1]]]

        # Fit the preprocessor and transform the training features
        train_validate_split[0], train_validate_split[2] = (
            preprocessor.fit_transform(
                train_validate_split[0], train_validate_split[2]))

        # Transform the validation features
        train_validate_split[1], train_validate_split[3] = (
            preprocessor.transform(
                train_validate_split[1], train_validate_split[3]))

        # Check if the model is a scikit-learn classifier
        if type(model_instance).__name__ in (
                'DecisionTreeClassifier', 'KNeighborsClassifier',
                'LogisticRegression', 'GaussianNB', 'RandomForestClassifier',
                'SVC'):

            # Generate a model clone
            model_clone = clone(model_instance)

            # Fit the model with the training split
            model_clone.fit(train_validate_split[0], train_validate_split[2])

        # Else, check if the model is a tensorflow model
        elif type(model_instance).__name__ == 'Functional':

            # Generate a model clone
            model_clone = clone_model(model_instance)

            # Compile the model
            model_clone.compile(
                optimizer=optimizer_map[hyperparameters['optimizer']](
                    hyperparameters['learning_rate']),
                loss=loss_map[hyperparameters['loss']]())

            # Fit the model with the training split
            model_clone.fit(
                train_validate_split[0],
                train_validate_split[2],
                batch_size=hyperparameters['batch_size'],
                epochs=hyperparameters['epochs'],
                verbose=0,
                callbacks=[
                    ReduceLROnPlateau(
                        monitor='loss', min_delta=0,
                        factor=hyperparameters['ReduceLROnPlateau_factor'],
                        patience=hyperparameters['ReduceLROnPlateau_patience'],
                        mode='min', verbose=0),
                    EarlyStopping(
                        monitor='loss', min_delta=0,
                        patience=hyperparameters['EarlyStopping_patience'],
                        mode='min', baseline=None, restore_best_weights=True,
                        verbose=0)
                    ],
                class_weight={
                    label: (2*len(train_validate_split[2])
                            / sum(train_validate_split[2] == label))
                    for label in (0, 1)}
                )

        # Compute the permutation importance for the validation split
        importance = permutation_importance(
            model_clone, train_validate_split[1], train_validate_split[3],
            scoring=score, n_repeats=number_of_repeats, random_state=42)

        return importance['importances'].T

    def score(model_instance, features, true_labels):
        """Score a model instance."""

        # Check if the model is a scikit-learn classifier
        if type(model_instance).__name__ in (
                'DecisionTreeClassifier', 'KNeighborsClassifier',
                'LogisticRegression', 'GaussianNB', 'RandomForestClassifier',
                'SVC'):

            # Check if the feature array has only a single row
            if features.shape[0] == 1:

                # Predict a single label value
                predicted_labels = model_instance.predict_proba(features)[0][1]

            else:

                # Predict an array with label values
                predicted_labels = model_instance.predict_proba(features)[:, 1]

        # Else, check if the model is a tensorflow model
        elif type(model_instance).__name__ == 'Functional':

            # Check if the feature array has only a single row
            if features.shape[0] == 1:

                # Predict a single label value
                predicted_labels = model_instance.predict(
                    features, verbose=0)[0][0]

            else:

                # Predict an array with label values
                predicted_labels = model_instance.predict(
                    features, verbose=0)[:, 0]

        return roc_auc_score(true_labels, predicted_labels)

    # Log a message about the permutation importance computation
    Datahub().logger.display_info("Computing permutation importances ...")

    # Initialize the data preprocessor
    preprocessor = DataPreprocessor(preprocessing_steps, verbose=False)

    # Compute the training permutation importance
    training_importance = permutation_importance(
        model_instance, *preprocessor.fit_transform(features, labels),
        scoring=score, n_repeats=number_of_repeats, random_state=42)

    # Compute the out-of-folds permutation importance
    fold_importances = tuple(map(compute_fold_importances, (
        (training_indices, validation_indices)
        for training_indices, validation_indices in (
                (where(oof_folds != number), where(oof_folds == number))
                for number in set(oof_folds)))))

    return {'Training': training_importance['importances'].T,
            'Out-of-folds': vstack(fold_importances)}
