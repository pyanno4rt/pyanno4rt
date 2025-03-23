"""Segment eigenvalues feature."""

# Author: Tim Ortkamp

# %% External package import

import jax.numpy as jnp

# %% Internal package import

from pyanno4rt.learning.features import RadiomicFeature

# %% Class definition


class SegmentEigenvalues(RadiomicFeature):
    """Segment eigenvalues feature class."""

    @staticmethod
    def compute(
            mask,
            spacing):
        """
        Compute the segment eigenvalues.

        Parameters
        ----------
        mask : ndarray
            Binary mask for the segment.

        spacing : ndarray
            Spacing of the dose grid.

        Returns
        -------
        object of class :class:`~jaxlib.xla_extension.ArrayImpl`
            Segment eigenvalues.

        object of class :class:`~jaxlib.xla_extension.ArrayImpl`
            Segment covariance matrix.
        """

        def compute_masked_mean(array):
            """Compute the array mean filtered by the mask."""

            return jnp.sum(array*mask) / jnp.sum(mask)

        def compute_masked_sum(array):
            """Compute the array sum filtered by the mask."""

            return jnp.sum(array*mask)

        # Determine the axis points
        points = jnp.meshgrid(
            jnp.array(range(mask.shape[1])),
            jnp.array(range(mask.shape[0])),
            jnp.array(range(mask.shape[2])))

        # Scale up with the grid spacings
        points_x, points_y, points_z = (
            point*space for point, space in zip(points, spacing))

        # Compute the masked means
        mean_x, mean_y, mean_z = tuple(
            compute_masked_mean(points)
            for points in (points_x, points_y, points_z))

        # Compute the covariance matrix terms
        covariance_terms = (
            (points_x-mean_x)**2,
            (points_x-mean_x) * (points_y-mean_y),
            (points_x-mean_x) * (points_z-mean_z),
            (points_x-mean_x) * (points_y-mean_y),
            (points_y-mean_y)**2,
            (points_y-mean_y) * (points_z-mean_z),
            (points_x-mean_x) * (points_z-mean_z),
            (points_y-mean_y) * (points_z-mean_z),
            (points_z-mean_z)**2)

        # Compute the masked sums
        matrix_elements = tuple(
            compute_masked_sum(term) for term in covariance_terms)

        # Reshape the matrix elements
        covariance_matrix = jnp.array(matrix_elements).reshape((3, 3))

        # Compute the eigenvalues
        eigenvalues, _ = jnp.linalg.eig(covariance_matrix)

        return eigenvalues.real, covariance_matrix
