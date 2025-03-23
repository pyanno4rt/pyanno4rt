"""Segment maximum eigenvalue feature test."""

# Authors: Moritz Müller, Tim Ortkamp

# %% External package import

from jax.numpy import array, identity, isclose
from pytest import mark

# %% Internal package import

from pyanno4rt.learning.features import SegmentEigenmax

# %% Test definition


# Define the argument sets
@mark.parametrize(
    'mask, spacing, expected',
    [(*(array([identity(3), identity(3), identity(3)]), (1, 1, 1)), 12)],
    ids=['case_A'])
def test_segment_eigenmax(mask, spacing, expected):
    """Test the 'SegmentEigenmax.compute' method."""

    # Assert the equality between actual and expected outcome
    assert isclose(SegmentEigenmax.compute(mask, spacing), expected, atol=1e-6)
