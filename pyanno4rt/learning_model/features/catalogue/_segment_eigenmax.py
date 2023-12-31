"""Segment maximum eigenvalue feature."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

import jax.numpy as jnp

# %% Internal package import

from pyanno4rt.learning_model.features.catalogue import RadiomicFeature
from pyanno4rt.learning_model.features.catalogue import SegmentEigenvalues

# %% Class definition


class SegmentEigenmax(RadiomicFeature):
    """Segment maximum eigenvalue feature class."""

    @staticmethod
    def compute(mask, spacing):
        """Compute the maximum eigenvalue."""
        # Compute the eigenvalues
        eigenvalues, _ = SegmentEigenvalues.compute(mask, spacing)

        return jnp.max(eigenvalues)
