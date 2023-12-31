"""Segment eigenvalues feature."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

import jax.numpy as jnp

# %% Internal package import

from pyanno4rt.learning_model.features.catalogue import RadiomicFeature

# %% Class definition


class SegmentEigenvalues(RadiomicFeature):
    """Segment eigenvalues feature class."""

    @staticmethod
    def compute(mask, spacing):
        """Compute all eigenvalues."""

        def compute_masked_mean(array):
            """Compute the mean of an array with a binary filter mask."""
            return jnp.sum((array*mask)) / jnp.sum(mask)

        def compute_masked_sum(array):
            """Compute the sum of an array with a binary filter mask."""
            return jnp.sum((array*mask))

        # Determine the axis points from a meshed grid
        points_x, points_y, points_z = jnp.meshgrid(
            jnp.array(range(mask.shape[0])),
            jnp.array(range(mask.shape[1])),
            jnp.array(range(mask.shape[2])))

        # Scale the axes points with the grid spacings
        points_x, points_y, points_z = (points_x*spacing[0],
                                        points_y*spacing[1],
                                        points_z*spacing[2])

        # Compute the means of the axis points
        mean_x, mean_y, mean_z = tuple(
            compute_masked_mean(points)
            for points in (points_x, points_y, points_z))

        # Compute the single terms in the covariance matrix function
        covariance_terms = (jnp.power(points_x-mean_x, 2),
                            (points_x-mean_x) * (points_y-mean_y),
                            (points_x-mean_x) * (points_z-mean_z),
                            (points_x-mean_x) * (points_y-mean_y),
                            jnp.power(points_y-mean_y, 2),
                            (points_y-mean_y) * (points_z-mean_z),
                            (points_x-mean_x) * (points_z-mean_z),
                            (points_y-mean_y) * (points_z-mean_z),
                            jnp.power(points_z-mean_z, 2))

        # Compute the masked versions of the covariance matrix terms
        matrix_elements = tuple(compute_masked_sum(term)
                                for term in covariance_terms)

        # Reshape the matrix elements to yield the covariance matrix
        covariance_matrix = jnp.array(matrix_elements).reshape((3, 3))

        # Compute the eigenvalues of the covariance matrix
        eigenvalues, _ = jnp.linalg.eig(covariance_matrix)

        return eigenvalues.real, covariance_matrix
