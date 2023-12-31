"""Segment compactness feature."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% Internal package import

from pyanno4rt.learning_model.features.catalogue import RadiomicFeature
from pyanno4rt.learning_model.features.catalogue import SegmentArea
from pyanno4rt.learning_model.features.catalogue import SegmentVolume

# %% Class definition


class SegmentCompactness(RadiomicFeature):
    """Segment compactness feature class."""

    @staticmethod
    def compute(mask, spacing):
        """Compute the compactness."""
        return (SegmentArea.compute(mask, spacing)
                / SegmentVolume.compute(mask, spacing))
