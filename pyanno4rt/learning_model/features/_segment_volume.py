"""Segment volume feature."""

# Author: Tim Ortkamp

# %% External package import

import jax.numpy as jnp

# %% Internal package import

from pyanno4rt.learning_model.features import RadiomicFeature

# %% Class definition


class SegmentVolume(RadiomicFeature):
    """Segment volume feature class."""

    @staticmethod
    def compute(
            mask,
            spacing):
        """
        Compute the segment volume.

        Parameters
        ----------
        mask : ndarray
            Binary mask for the segment.

        spacing : ndarray
            Spacing of the dose grid.

        Returns
        -------
        object of class :class:`~jaxlib.xla_extension.ArrayImpl`
            Segment volume.
        """

        return jnp.sum(mask) * jnp.prod(spacing)
