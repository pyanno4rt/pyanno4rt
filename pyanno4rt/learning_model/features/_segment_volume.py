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
    def compute(mask, spacing):
        """Compute the volume."""
        return jnp.sum(mask) * jnp.prod(spacing)
