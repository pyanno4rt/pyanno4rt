"""Segment eccentricity feature."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

import jax.numpy as jnp

# %% Internal package import

from pyanno4rt.learning_model.features.catalogue import RadiomicFeature
from pyanno4rt.learning_model.features.catalogue import SegmentEigenvalues

# %% Class definition


class SegmentEccentricity(RadiomicFeature):
    """Segment eccentricity feature class."""

    @staticmethod
    def compute(mask, spacing):
        """Compute the eccentricity."""
        # Compute the eigenvalues
        eigenvalues, _ = SegmentEigenvalues.compute(mask, spacing)

        return 1 - jnp.sqrt(jnp.min(eigenvalues)/jnp.max(eigenvalues))
