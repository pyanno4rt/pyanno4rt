"""Segment maximum eigenvalue feature."""

# Author: Tim Ortkamp

# %% External package import

from jax.numpy import max as jmax

# %% Internal package import

from pyanno4rt.learning.features import RadiomicFeature
from pyanno4rt.learning.features import SegmentEigenvalues

# %% Class definition


class SegmentEigenmax(RadiomicFeature):
    """Segment maximum eigenvalue feature class."""

    @staticmethod
    def compute(
            mask,
            resolution):
        """
        Compute the maximum segment eigenvalue.

        Parameters
        ----------
        mask : ndarray
            Binary mask for the segment.

        resolution : ndarray
            Grid resolution (in mm).

        Returns
        -------
        object of class :class:`~jaxlib.xla_extension.ArrayImpl`
            Maximum segment eigenvalue.
        """

        # Compute the eigenvalues
        eigenvalues, _ = SegmentEigenvalues.compute(mask, resolution)

        return jmax(eigenvalues)
