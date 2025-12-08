"""Data medoid initialization."""

# Author: Tim Ortkamp

# %% External package import

from random import sample

from functools import partial
from math import inf
from numpy import (
    array, argmin, ceil, concatenate, divide, floor, log, mean, median, ones,
    std, where, zeros)
from numpy.linalg import norm
from scipy.optimize import minimize, minimize_scalar
from scipy.sparse import hstack as shstack

# %% Internal package import

from pyanno4rt.logging import get_logger

# %% Class definition


class DataMedoidInitializer():
    """
    Data Medoid initialization class.

    This class provides methods to initialize the fluence vector with respect \
    to data medoid points.

    Parameters
    ----------
    initial_fluence_vector: None or list
        User-defined initial fluence vector for the optimization problem.

    Attributes
    ----------
    initial_fluence_vector : None or list
        See 'Parameters'.
    """

    def __init__(
            self,
            initial_fluence_vector=None):

        # Log a message about the initialization of the class
        get_logger().info("Initializing data medoid strategy ...")

        # Get the initial fluence
        self.initial_fluence_vector = initial_fluence_vector

    def run(
            self,
            handlers):
        """
        Initialize the fluence vector with respect to target coverage.

        Parameters
        ----------
        handlers : dict
            Dictionary with the handlers (patient, plan, dose, data models).

        Returns
        -------
        ndarray
            Initial fluence vector.
        """

        # Check if an initial fluence vector has been provided
        if self.initial_fluence_vector is not None:

            # Log a message about returning the user-defined vector
            get_logger().info(
                "Falling back to user-defined initial fluence vector ...")

            return array(self.initial_fluence_vector)

        # Log a message about the initialization
        get_logger().info(
            "Initializing fluence vector with respect to data medoid points "
            "...")

        # Get the datasets and feature maps from the datahub
        datasets = handlers['data_model_handler'].datasets
        feature_maps = handlers['data_model_handler'].feature_maps

        # Check if no datasets have been provided
        if datasets is None:

            # Log a message about falling back to target coverage strategy
            get_logger().info(
                "No datasets have been provided - falling back to target "
                "coverage initialization strategy ...")

            # Import the initializers dynamically
            from pyanno4rt.optimization._maps import INITIALIZERS

            return INITIALIZERS['target-coverage']().run(handlers)

        def get_standardized_features(key):
            """Get the standardized dose features."""

            # Get the columns of the dosiomic features
            columns = [
                index for index, feature in enumerate(feature_maps[key])
                if feature_maps[key][feature]['class'] == 'Dosiomics']

            # Extract the dosiomic feature values
            values = datasets[key]['feature_values'][:, columns]

            # Calculate the mean and standard deviation per column
            means = mean(values, axis=0)
            deviations = std(values, axis=0)

            return divide(values-means, deviations), means, deviations

        def get_data_medoid(dataset):
            """
            Get the medoid from a dataset via Correlated Sequential Halving.

            Adapted from Baharav & Tse (2019): https://arxiv.org/abs/1906.04356
            """

            def pull_arms(index_set, number_of_pulls):
                """
                Pull the arms of the multi-armed bandit to update the \
                scores and the pull history.
                """

                # Uniformly sample arms
                random_arms = sample(range(number_of_samples), number_of_pulls)

                # Estimate the correlated distances
                estimates = array([
                    mean(norm(dataset[random_arms, :] - dataset[index, :]))
                    for index in index_set])

                # Update the scores taking the pull history into account
                scores[index_set] = (
                    (pull_history[index_set]*scores[index_set]
                     + number_of_pulls*estimates) /
                    (pull_history[index_set]+number_of_pulls))

                # Check if all bandit arms are pulled
                if number_of_pulls == number_of_samples:

                    # Update the scores by the exact estimates
                    scores[index_set] = estimates

                # Update the pull history
                pull_history[index_set] += number_of_pulls

                return scores[index_set]

            # Get the number of samples in the dataset
            number_of_samples = dataset.shape[0]

            # Initialize the index set
            index_set = array(range(number_of_samples))

            # Initialize the scores for all samples
            scores = zeros(number_of_samples)

            # Initialize the multi-armed bandit pull history
            pull_history = zeros(number_of_samples, dtype=int)

            # Loop while the cardinality of the index set exceeds one
            while len(index_set) > 1:

                # Get the number of pulls on the bandit
                number_of_pulls = int(
                    min(max(1, floor(30*number_of_samples/(len(index_set)*ceil(
                        log(number_of_samples))))), number_of_samples))

                # Get the updated score set
                score_set = pull_arms(index_set, number_of_pulls)

                # Check if all bandit arms are pulled
                if number_of_pulls == number_of_samples:

                    # Return the sample assigned with the lowest score
                    return dataset[index_set[argmin(score_set)]]

                # Reduce the index set by eliminating the worse half of arms
                index_set = index_set[
                    where(score_set <= median(score_set))[0]]

            return dataset[index_set, :]

        def optimize_fluence(medoids, means, deviations):
            """Optimize the fluence vector with respect to the data medoids."""

            # Get the degrees of freedom from the datahub
            degrees_of_freedom = handlers['dose_handler'].degrees_of_freedom

            def precompute(fluence, factor):
                """Precompute the features and the segment doses/names."""

                # Get the dose vector from the fluence
                dose = (
                    handlers['dose_handler'].dose_influence_matrix
                    @ (fluence*factor))

                # Get the segments across the feature maps
                segments = tuple(set(
                    feature_map[feature]['segment'] for feature in feature_map)
                    for feature_map in feature_maps.values())

                # Get the dose vectors for the segments
                doses = tuple(
                    (dose[handlers['patient_handler'].segmentation[subsegment][
                        'resized_indices']] for subsegment in segment)
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
                     - reference)**2)

            def gradient(fluence):
                """Compute the squared L2 objective gradient."""

                # Get the feature vector, doses and segments
                features, doses, segments = precompute(fluence, factor=1)

                # Get the dose gradient of the features
                feature_gradient = shstack(
                    [calculator.gradientize(dose, segment).T
                     for calculator, dose, segment in zip(
                             calculators, doses, segments)])

                return ((
                    2*(divide(features-means, deviations)-reference)
                    * (1/deviations)*feature_gradient.T)
                    @ handlers['dose_handler'].dose_influence_matrix)

            # Concatenate the data medoids, mean values and standard deviations
            reference = concatenate(list(medoids))
            means = concatenate(means)
            deviations = concatenate(deviations)

            # Get the feature maps reduced to dosiomic features
            dose_feature_maps = tuple(
                {key: value for key, value in feature_map.items()
                 if feature_map[key]['class'] == 'Dosiomics'}
                for feature_map in feature_maps.values())

            # Get the feature calculator for each dosiomic subset
            calculators = tuple(FeatureCalculator(
                write_features=False, verbose=False).add_feature_map(
                    dose_map, return_self=True)
                for dose_map in dose_feature_maps)

            # Optimize the fluence under homogeneity condition
            factor_result = minimize_scalar(
                fun=partial(objective, ones(degrees_of_freedom)),
                bounds=(0, 100),
                method='bounded',
                options={
                    'disp': False,
                    'maxiter': 1000})

            # Optimize the fluence under heterogeneity condition
            fluence_result = minimize(
                x0=[factor_result.x]*degrees_of_freedom,
                fun=partial(objective, factor=1),
                jac=gradient,
                bounds=zip([0]*degrees_of_freedom, [inf]*degrees_of_freedom),
                tol=0.001,
                method='L-BFGS-B',
                callback=None,
                options={
                    'disp': False,
                    'ftol': 0.001,
                    'maxiter': 1000,
                    'maxls': 20})

            return fluence_result.x

        # Get the standardized datasets, mean vectors and standard deviations
        standardized_data, means, deviations = zip(*map(
            get_standardized_features, datasets))

        # Get the data medoids
        medoids = map(get_data_medoid, standardized_data)

        # Optimize the fluence by reconstructing the medoids
        initial_fluence = optimize_fluence(medoids, means, deviations)

        return initial_fluence
