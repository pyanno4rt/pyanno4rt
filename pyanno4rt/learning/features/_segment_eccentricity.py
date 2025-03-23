"""Segment eccentricity feature."""

# Author: Tim Ortkamp

# %% External package import

import jax.numpy as jnp

# %% Internal package import

from pyanno4rt.learning.features import RadiomicFeature
from pyanno4rt.learning.features import SegmentEigenvalues

# %% Class definition


class SegmentEccentricity(RadiomicFeature):
    """Segment eccentricity feature class."""

    @staticmethod
    def compute(
            mask,
            spacing):
        """
        Compute the segment eccentricity.

        Parameters
        ----------
        mask : ndarray
            Binary mask for the segment.

        spacing : ndarray
            Spacing of the dose grid.

        Returns
        -------
        object of class :class:`~jaxlib.xla_extension.ArrayImpl`
            Segment eccentricity.
        """

        # Compute the eigenvalues
        eigenvalues, _ = SegmentEigenvalues.compute(mask, spacing)

        return 1 - jnp.sqrt(jnp.min(eigenvalues)/jnp.max(eigenvalues))
