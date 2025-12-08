"""Segment eigenvalues feature."""

# Author: Tim Ortkamp

# %% External package import

from jax.numpy import array, linalg, meshgrid
from jax.numpy import sum as jsum

# %% Internal package import

from pyanno4rt.learning.features import RadiomicFeature

# %% Class definition


class SegmentEigenvalues(RadiomicFeature):
    """Segment eigenvalues feature class."""

    @staticmethod
    def compute(
            mask,
            resolution):
        """
        Compute the segment eigenvalues.

        Parameters
        ----------
        mask : ndarray
            Binary mask for the segment.

        resolution : ndarray
            Grid resolution (in mm).

        Returns
        -------
        object of class :class:`~jaxlib.xla_extension.ArrayImpl`
            Segment eigenvalues.

        object of class :class:`~jaxlib.xla_extension.ArrayImpl`
            Segment covariance matrix.
        """

        # Determine the axis points
        points = meshgrid(
            array(range(mask.shape[1])),
            array(range(mask.shape[0])),
            array(range(mask.shape[2])))

        # Scale up with the grid resolution
        points_x, points_y, points_z = (
            point*spacing for point, spacing in zip(points, resolution))

        # Compute the masked means
        mean_x, mean_y, mean_z = tuple(
            jsum(points*mask) / jsum(mask)
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
        matrix_elements = tuple(jsum(term*mask) for term in covariance_terms)

        # Reshape the matrix elements
        covariance_matrix = array(matrix_elements).reshape((3, 3))

        # Compute the eigenvalues
        eigenvalues, _ = linalg.eig(covariance_matrix)

        return eigenvalues.real, covariance_matrix
