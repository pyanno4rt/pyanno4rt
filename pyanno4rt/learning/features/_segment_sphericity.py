"""Segment sphericity feature."""

# Author: Tim Ortkamp

# %% External package import

from math import pi

from jax.numpy import power

# %% Internal package import

from pyanno4rt.learning.features import RadiomicFeature
from pyanno4rt.learning.features import SegmentArea
from pyanno4rt.learning.features import SegmentVolume

# %% Class definition


class SegmentSphericity(RadiomicFeature):
    """Segment sphericity feature class."""

    @staticmethod
    def compute(
            mask,
            resolution):
        """
        Compute the segment sphericity.

        Parameters
        ----------
        mask : ndarray
            Binary mask for the segment.

        resolution : ndarray
            Grid resolution (in mm).

        Returns
        -------
        object of class :class:`~jaxlib.xla_extension.ArrayImpl`
            Segment sphericity.
        """

        # Compute the segment area
        area = SegmentArea.compute(mask, resolution)

        # Compute the segment volume
        volume = SegmentVolume.compute(mask, resolution)

        return power(pi, 1/3) * power(6*volume, 2/3) / area
