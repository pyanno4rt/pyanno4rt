"""Segment volume feature."""

# Author: Tim Ortkamp

# %% External package import

from jax.numpy import prod
from jax.numpy import sum as jsum

# %% Internal package import

from pyanno4rt.learning.features import RadiomicFeature

# %% Class definition


class SegmentVolume(RadiomicFeature):
    """Segment volume feature class."""

    @staticmethod
    def compute(
            mask,
            resolution):
        """
        Compute the segment volume.

        Parameters
        ----------
        mask : ndarray
            Binary mask for the segment.

        resolution : ndarray
            Grid resolution (in mm).

        Returns
        -------
        object of class :class:`~jaxlib.xla_extension.ArrayImpl`
            Segment volume.
        """

        return jsum(mask) * prod(resolution)
