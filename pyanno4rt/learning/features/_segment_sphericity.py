"""Segment sphericity feature."""

# Author: Tim Ortkamp

# %% External package import

from math import pi

import jax.numpy as jnp

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
            spacing):
        """
        Compute the segment sphericity.

        Parameters
        ----------
        mask : ndarray
            Binary mask for the segment.

        spacing : ndarray
            Spacing of the dose grid.

        Returns
        -------
        object of class :class:`~jaxlib.xla_extension.ArrayImpl`
            Segment sphericity.
        """

        # Compute the segment area
        area = SegmentArea.compute(mask, spacing)

        # Compute the segment volume
        volume = SegmentVolume.compute(mask, spacing)

        return jnp.power(pi, 1/3) * jnp.power(6*volume, 2/3) / area
