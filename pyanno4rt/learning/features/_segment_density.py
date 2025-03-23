"""Segment density feature."""

# Author: Tim Ortkamp

# %% External package import

import jax.numpy as jnp

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
            spacing):
        """
        Compute the segment density.

        Parameters
        ----------
        mask : ndarray
            Binary mask for the segment.

        spacing : ndarray
            Spacing of the dose grid.

        Returns
        -------
        object of class :class:`~jaxlib.xla_extension.ArrayImpl`
            Segment density.
        """

        # Compute the covariance matrix
        _, covariance_matrix = SegmentEigenvalues.compute(mask, spacing)

        # Compute the segment volume
        volume = SegmentVolume.compute(mask, spacing)

        return volume**(1/3) / jnp.trace(covariance_matrix)
