"""Data medoid initialization."""

# Author: Tim Ortkamp

# %% External package import

from random import sample

from functools import partial
from math import inf
from numpy import (
    array, argmin, ceil, concatenate, divide, floor, hstack, log, mean, median,
    ones, std, where, zeros)
from numpy.linalg import norm
from scipy.optimize import minimize, minimize_scalar

# %% Internal package import

from pyanno4rt.learning.features import FeatureCalculator
from pyanno4rt.logging import get_logger

# %% Class definition


class DataMedoidInitializer():
    """
    Data medoid initialization class.

    This class provides methods to initialize the fluence vector with respect \
    to data medoid points.

    Parameters
    ----------
    initial_fluence: None or list
        Initial fluence vector.

    Attributes
    ----------
    initial_fluence : None or list
        See 'Parameters'.
    """

    def __init__(
            self,
            initial_fluence=None):

        # Log a message about the initialization of the class
        get_logger().info("Initializing data medoid strategy ...")

        # Get the initial fluence
        self.initial_fluence = initial_fluence

    def run(
            self,
            handlers):
        """
        Initialize the fluence vector with respect to data medoid points.

        Parameters
        ----------
        handlers : dict
            Dictionary with the handlers (patient, plan, dose, data models).

        Returns
        -------
        ndarray
            Initial fluence vector.
        """

        def standardize(dataset, feature_map):
            """Get the standardized dose features."""

            # Get the columns of the dosiomic features
            columns = [
                index for index, feature in enumerate(feature_map)
                if feature_map[feature]['class'] == 'Dosiomics']

            # Extract the dosiomic feature values
            features = dataset.feature_values[:, columns]

            # Calculate the column-wise mean and standard deviation
            means = mean(features, axis=0)
            deviations = std(features, axis=0)

            return divide(features-means, deviations), means, deviations

        def get_data_medoid(features):
            """
            Get the medoid from a dataset via Correlated Sequential Halving.

            Adapted from Baharav & Tse (2019): https://arxiv.org/abs/1906.04356
            """

            def pull_arms(indices, number_of_pulls):
                """
                Pull the arms of the multi-armed bandit to update the \
                scores and the pull history.
                """

                # Uniformly sample arms
                random_arms = sample(range(number_of_samples), number_of_pulls)

                # Estimate the correlated distances
                corr_distances = array([
                    mean(norm(features[random_arms, :] - features[index, :]))
                    for index in indices])

                # Update the scores by the pull history
                scores[indices] = (
                    (pull_history[indices]*scores[indices]
                     + number_of_pulls*corr_distances) /
                    (pull_history[indices]+number_of_pulls))

                # Check if all bandit arms are pulled
                if number_of_pulls == number_of_samples:

                    # Update the scores
                    scores[indices] = corr_distances

                # Update the pull history
                pull_history[indices] += number_of_pulls

                return scores[indices]

            # Get the number of samples
            number_of_samples = features.shape[0]

            # Initialize the indices
            indices = array(range(number_of_samples))

            # Initialize the scores
            scores = zeros(number_of_samples)

            # Initialize the multi-armed bandit pull history
            pull_history = zeros(number_of_samples, dtype=int)

            # Loop while the cardinality of the indices exceeds one
            while len(indices) > 1:

                # Get the number of pulls on the bandit
                number_of_pulls = int(
                    min(max(1, floor(30*number_of_samples/(len(indices)*ceil(
                        log(number_of_samples))))), number_of_samples))

                # Get the updated score set
                score_set = pull_arms(indices, number_of_pulls)

                # Check if all bandit arms are pulled
                if number_of_pulls == number_of_samples:

                    # Return the sample assigned with the lowest score
                    return features[indices[argmin(score_set)]]

                # Reduce the indices by eliminating the worse half of arms
                indices = indices[where(score_set <= median(score_set))[0]]

            return features[indices, :]

        def approximate_fluence(medoids, means, deviations):
            """Approximate the fluence with respect to the data medoids."""

            def precompute(fluence, factor):
                """Precompute the features, doses and segment names."""

                # Get the segments across the feature maps
                segments = tuple(tuple(set(feature['segment']
                    for feature in feature_map.values()))
                    for feature_map in feature_maps)

                # Calculate the dose from the fluence
                dose = (
                    handlers['dose_handler'].dose_influence_matrix
                    @ (fluence*factor))

                # Get the segment doses
                doses = tuple(
                    [dose[handlers['patient_handler'].segmentation[subsegment][
                        'resized_indices']] for subsegment in segment]
                    for segment in segments)

                # Calculate the dosiomic feature values
                features = concatenate(
                    [calculator.featurize(dose, segment, no_cache=True).T
                     for calculator, dose, segment in zip(
                             calculators, doses, segments)]).reshape(-1)

                return features, doses, segments

            def objective(fluence, factor):
                """Compute the squared L2 objective value."""

                return sum(
                    (divide(precompute(fluence, factor)[0]-means, deviations)
                     - medoids)**2)

            def gradient(fluence):
                """Compute the squared L2 objective gradient."""

                # Get the features, doses and segments
                features, doses, segments = precompute(fluence, factor=1)

                # Get the dose gradient of the features
                feature_gradient = hstack(
                    [calculator.gradientize(dose, segment).T
                     for calculator, dose, segment in zip(
                             calculators, doses, segments)])

                # Get the segment indices
                indices = concatenate(tuple(
                    [handlers['patient_handler'].segmentation[subsegment][
                        'resized_indices'] for subsegment in segment]
                    for segment in segments)).reshape(-1)

                # Get the dose-influence matrix
                dose_matrix = handlers['dose_handler'].dose_influence_matrix

                return (
                    dose_matrix[indices, :].T
                    @ (2*(divide(features-means, deviations)-medoids)
                       *(1/deviations)*feature_gradient)).sum(axis=1)

            # Concatenate the data medoids, means and standard deviations
            medoids = concatenate(medoids)
            means = concatenate(means)
            deviations = concatenate(deviations)

            # Get the feature maps for the dosiomic features
            dose_feature_maps = (
                {key: value for key, value in feature_map.items()
                 if feature_map[key]['class'] == 'Dosiomics'}
                for feature_map in feature_maps)

            # Get the corresponding feature calculators
            calculators = tuple(
                FeatureCalculator(handlers).set_mapping(
                    dose_map, return_self=True, verbose=False)
                for dose_map in dose_feature_maps)

            # Get the degrees of freedom
            degrees_of_freedom = handlers['dose_handler'].degrees_of_freedom

            # Approximate the fluence under homogeneity condition
            uniform = minimize_scalar(
                fun=partial(objective, ones(degrees_of_freedom)),
                bounds=(0, 1000),
                method='bounded',
                options={
                    'disp': False,
                    'maxiter': 1000})

            # Approximate the fluence under heterogeneity condition
            result = minimize(
                x0=[uniform.x]*degrees_of_freedom,
                fun=partial(objective, factor=1),
                jac=gradient,
                bounds=zip([0]*degrees_of_freedom, [inf]*degrees_of_freedom),
                tol=1e-4,
                method='L-BFGS-B',
                callback=None,
                options={
                    'disp': False,
                    'ftol': 1e-4,
                    'maxiter': 1000,
                    'maxls': 20})

            return result.x

        # Check if an initial fluence vector has been provided
        if self.initial_fluence is not None:

            # Log a message about falling back to warm-start strategy
            get_logger().warning(
                "User has provided an initial fluence vector - falling back "
                "to warm-start strategy ...")

            return array(self.initial_fluence)

        # Log a message about the initialization
        get_logger().info(
            "Initializing fluence vector with respect to data medoid points "
            "...")

        # Get the datasets and feature maps
        datasets, feature_maps = zip(*(
            (model.dataset, model.feature_calculator.feature_map)
            for model in handlers['data_model_handler'].models))

        # Check if no datasets have been provided
        if len(datasets) == 0:

            # Log a message about falling back to target coverage strategy
            get_logger().info(
                "No datasets have been provided - falling back to target "
                "coverage initialization strategy ...")

            # Import the initializers dynamically
            from pyanno4rt.optimization._maps import INITIALIZERS

            return INITIALIZERS['target-coverage']().run(handlers)

        # Get the standardized datasets, mean vectors and standard deviations
        standardized_data, means, deviations = zip(*map(
            standardize, datasets, feature_maps))

        # Get the data medoids
        medoids = tuple(map(get_data_medoid, standardized_data))

        # Approximate the initial fluence by reconstructing the medoids
        initial_fluence = approximate_fluence(medoids, means, deviations)

        return initial_fluence
