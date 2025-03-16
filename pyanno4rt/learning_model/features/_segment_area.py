"""Segment area feature."""

# Author: Tim Ortkamp

# %% External package import

import jax.numpy as jnp

# %% Internal package import

from pyanno4rt.learning_model.features import RadiomicFeature

# %% Class definition


class SegmentArea(RadiomicFeature):
    """Segment area feature class."""

    @staticmethod
    def compute(
            mask,
            spacing):
        """
        Compute the segment area.

        Parameters
        ----------
        mask : ndarray
            Binary mask for the segment.

        spacing : ndarray
            Spacing of the dose grid.

        Returns
        -------
        object of class :class:`~jaxlib.xla_extension.ArrayImpl`
            Segment area.
        """

        def compute_directional_area(element):
            """Compute the area in one direction."""

            # Compute the terms
            terms = jnp.array([
                jnp.sum(jnp.sum(value[2])) if value[0] in (0, element[0]-1)
                else jnp.sum(jnp.sum(abs(value[1]-value[2])))
                for value in element[1]])

            return jnp.sum(terms) * element[2]

        # Set the input elements
        elements = (
            (mask.shape[0], ((i, mask[i+1, :, :], mask[i, :, :])
             for i in range(mask.shape[0]-1)), spacing[1]*spacing[2]),
            (mask.shape[1], ((j, mask[:, j+1, :], mask[:, j, :])
             for j in range(mask.shape[1]-1)), spacing[0]*spacing[2]),
            (mask.shape[2], ((k, mask[:, :, k+1], mask[:, :, k])
             for k in range(mask.shape[2]-1)), spacing[0]*spacing[1]))

        return jnp.sum(jnp.array([
            compute_directional_area(element) for element in elements]))
