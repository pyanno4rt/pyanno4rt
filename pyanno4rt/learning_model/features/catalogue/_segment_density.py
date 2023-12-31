"""Segment density feature."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

import jax.numpy as jnp

# %% Internal package import

from pyanno4rt.learning_model.features.catalogue import RadiomicFeature
from pyanno4rt.learning_model.features.catalogue import SegmentEigenvalues
from pyanno4rt.learning_model.features.catalogue import SegmentVolume

# %% Class definition


class SegmentDensity(RadiomicFeature):
    """Segment density feature class."""

    @staticmethod
    def compute(mask, spacing):
        """Compute the density."""
        # Compute the covariance matrix
        _, covariance_matrix = SegmentEigenvalues.compute(mask, spacing)

        # Compute the segment volume
        volume = SegmentVolume.compute(mask, spacing)

        return jnp.power(volume, 1/3) / jnp.trace(covariance_matrix)
