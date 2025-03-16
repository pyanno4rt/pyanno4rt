"""Segment maximum eigenvalue feature."""

# Author: Tim Ortkamp

# %% External package import

import jax.numpy as jnp

# %% Internal package import

from pyanno4rt.learning_model.features import RadiomicFeature
from pyanno4rt.learning_model.features import SegmentEigenvalues

# %% Class definition


class SegmentEigenmax(RadiomicFeature):
    """Segment maximum eigenvalue feature class."""

    @staticmethod
    def compute(
            mask,
            spacing):
        """
        Compute the maximum segment eigenvalue.

        Parameters
        ----------
        mask : ndarray
            Binary mask for the segment.

        spacing : ndarray
            Spacing of the dose grid.

        Returns
        -------
        object of class :class:`~jaxlib.xla_extension.ArrayImpl`
            Maximum segment eigenvalue.
        """

        # Compute the eigenvalues
        eigenvalues, _ = SegmentEigenvalues.compute(mask, spacing)

        return jnp.max(eigenvalues)
