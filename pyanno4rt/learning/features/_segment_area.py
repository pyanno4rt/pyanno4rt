"""Segment area feature."""

# Author: Tim Ortkamp

# %% External package import

from jax.numpy import array
from jax.numpy import sum as jsum

# %% Internal package import

from pyanno4rt.learning.features import RadiomicFeature

# %% Class definition


class SegmentArea(RadiomicFeature):
    """Segment area feature class."""

    @staticmethod
    def compute(
            mask,
            resolution):
        """
        Compute the segment area.

        Parameters
        ----------
        mask : ndarray
            Binary mask for the segment.

        resolution : ndarray
            Grid resolution (in mm).

        Returns
        -------
        object of class :class:`~jaxlib.xla_extension.ArrayImpl`
            Segment area.
        """

        def get_planar_area(inputs):
            """Get the planar area."""

            # Compute the terms
            terms = array([
                jsum(jsum(value[2])) if value[0] in (0, inputs[0]-1)
                else jsum(jsum(abs(value[1]-value[2])))
                for value in inputs[1]])

            return jsum(terms) * inputs[2]

        # Set the input elements
        inputs = (
            (mask.shape[0], ((i, mask[i+1, :, :], mask[i, :, :])
             for i in range(mask.shape[0]-1)), resolution[1]*resolution[2]),
            (mask.shape[1], ((j, mask[:, j+1, :], mask[:, j, :])
             for j in range(mask.shape[1]-1)), resolution[0]*resolution[2]),
            (mask.shape[2], ((k, mask[:, :, k+1], mask[:, :, k])
             for k in range(mask.shape[2]-1)), resolution[0]*resolution[1]))

        return jsum(array([get_planar_area(element) for element in inputs]))
