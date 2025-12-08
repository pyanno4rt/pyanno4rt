"""Segment middle eigenvalue feature."""

# Author: Tim Ortkamp

# %% External package import

from jax.numpy import sort

# %% Internal package import

from pyanno4rt.learning.features import RadiomicFeature
from pyanno4rt.learning.features import SegmentEigenvalues

# %% Class definition


class SegmentEigenmid(RadiomicFeature):
    """Segment middle eigenvalue feature class."""

    @staticmethod
    def compute(
            mask,
            resolution):
        """
        Compute the middle segment eigenvalue.

        Parameters
        ----------
        mask : ndarray
            Binary mask for the segment.

        resolution : ndarray
            Grid resolution (in mm).

        Returns
        -------
        object of class :class:`~jaxlib.xla_extension.ArrayImpl`
            Middle segment eigenvalue.
        """

        # Compute the eigenvalues
        eigenvalues, _ = SegmentEigenvalues.compute(mask, resolution)

        return sort(eigenvalues)[1]
