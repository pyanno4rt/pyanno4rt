"""Segment compactness feature."""

# Author: Tim Ortkamp

# %% Internal package import

from pyanno4rt.learning.features import RadiomicFeature
from pyanno4rt.learning.features import SegmentArea
from pyanno4rt.learning.features import SegmentVolume

# %% Class definition


class SegmentCompactness(RadiomicFeature):
    """Segment compactness feature class."""

    @staticmethod
    def compute(
            mask,
            resolution):
        """
        Compute the segment compactness.

        Parameters
        ----------
        mask : ndarray
            Binary mask for the segment.

        resolution : ndarray
            Grid resolution (in mm).

        Returns
        -------
        object of class :class:`~jaxlib.xla_extension.ArrayImpl`
            Segment compactness.
        """

        return (
            SegmentArea.compute(mask, resolution)
            / SegmentVolume.compute(mask, resolution))
