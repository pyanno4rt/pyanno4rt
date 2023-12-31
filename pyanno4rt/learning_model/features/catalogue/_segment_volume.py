"""Segment volume feature."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

import jax.numpy as jnp

# %% Internal package import

from pyanno4rt.learning_model.features.catalogue import RadiomicFeature

# %% Class definition


class SegmentVolume(RadiomicFeature):
    """Segment volume feature class."""

    @staticmethod
    def compute(mask, spacing):
        """Compute the volume."""
        return jnp.sum(mask) * jnp.prod(spacing)
