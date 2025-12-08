"""Segment eccentricity feature."""

# Author: Tim Ortkamp

# %% External package import

from jax.numpy import sqrt
from jax.numpy import min as jmin
from jax.numpy import max as jmax

# %% Internal package import

from pyanno4rt.learning.features import RadiomicFeature
from pyanno4rt.learning.features import SegmentEigenvalues

# %% Class definition


class SegmentEccentricity(RadiomicFeature):
    """Segment eccentricity feature class."""

    @staticmethod
    def compute(
            mask,
            resolution):
        """
        Compute the segment eccentricity.

        Parameters
        ----------
        mask : ndarray
            Binary mask for the segment.

        resolution : ndarray
            Grid resolution (in mm).

        Returns
        -------
        object of class :class:`~jaxlib.xla_extension.ArrayImpl`
            Segment eccentricity.
        """

        # Compute the eigenvalues
        eigenvalues, _ = SegmentEigenvalues.compute(mask, resolution)

        return 1 - sqrt(jmin(eigenvalues)/jmax(eigenvalues))
