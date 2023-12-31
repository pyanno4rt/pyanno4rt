"""Segment sphericity feature."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from math import pi
import jax.numpy as jnp

# %% Internal package import

from pyanno4rt.learning_model.features.catalogue import RadiomicFeature
from pyanno4rt.learning_model.features.catalogue import SegmentArea
from pyanno4rt.learning_model.features.catalogue import SegmentVolume

# %% Class definition


class SegmentSphericity(RadiomicFeature):
    """Segment sphericity feature class."""

    @staticmethod
    def compute(mask, spacing):
        """Compute the sphericity."""
        # Compute the segment area
        area = SegmentArea.compute(mask, spacing)

        # Compute the segment volume
        volume = SegmentVolume.compute(mask, spacing)

        return jnp.power(pi, 1/3) * jnp.power(6*volume, 2/3) / area
