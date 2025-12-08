"""Segment density feature."""

# Author: Tim Ortkamp

# %% External package import

from jax.numpy import trace

# %% Internal package import

from pyanno4rt.learning.features import RadiomicFeature
from pyanno4rt.learning.features import SegmentEigenvalues
from pyanno4rt.learning.features import SegmentVolume

# %% Class definition


class SegmentDensity(RadiomicFeature):
    """Segment density feature class."""

    @staticmethod
    def compute(
            mask,
            resolution):
        """
        Compute the segment density.

        Parameters
        ----------
        mask : ndarray
            Binary mask for the segment.

        resolution : ndarray
            Grid resolution (in mm).

        Returns
        -------
        object of class :class:`~jaxlib.xla_extension.ArrayImpl`
            Segment density.
        """

        # Compute the covariance matrix
        _, covariance_matrix = SegmentEigenvalues.compute(mask, resolution)

        # Compute the segment volume
        volume = SegmentVolume.compute(mask, resolution)

        return volume**(1/3) / trace(covariance_matrix)
