"""Segment minimum eigenvalue feature."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

import jax.numpy as jnp

# %% Internal package import

from pyanno4rt.learning_model.features.catalogue import RadiomicFeature
from pyanno4rt.learning_model.features.catalogue import SegmentEigenvalues

# %% Class definition


class SegmentEigenmin(RadiomicFeature):
    """Segment minimum eigenvalue feature class."""

    @staticmethod
    def compute(mask, spacing):
        """Compute the minimum eigenvalue."""
        # Compute the eigenvalues
        eigenvalues, _ = SegmentEigenvalues.compute(mask, spacing)

        return jnp.min(eigenvalues)
