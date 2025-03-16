"""Segment compactness feature."""

# Author: Tim Ortkamp

# %% Internal package import

from pyanno4rt.learning_model.features import RadiomicFeature
from pyanno4rt.learning_model.features import SegmentArea
from pyanno4rt.learning_model.features import SegmentVolume

# %% Class definition


class SegmentCompactness(RadiomicFeature):
    """Segment compactness feature class."""

    @staticmethod
    def compute(
            mask,
            spacing):
        """
        Compute the segment compactness.

        Parameters
        ----------
        mask : ndarray
            Binary mask for the segment.

        spacing : ndarray
            Spacing of the dose grid.

        Returns
        -------
        object of class :class:`~jaxlib.xla_extension.ArrayImpl`
            Segment compactness.
        """
        return (
            SegmentArea.compute(mask, spacing)
            / SegmentVolume.compute(mask, spacing))
