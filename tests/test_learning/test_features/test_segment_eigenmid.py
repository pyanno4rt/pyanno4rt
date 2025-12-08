"""Segment middle eigenvalue feature test."""

# Authors: Moritz Müller, Tim Ortkamp

# %% External package import

from jax.numpy import array, identity, isclose
from pytest import mark

# %% Internal package import

from pyanno4rt.learning.features import SegmentEigenmid

# %% Test definition


# Define the argument sets
@mark.parametrize(
    'mask, resolution, expected',
    [(*(array([identity(3), identity(3), identity(3)]), (1, 1, 1)), 6)],
    ids=['case_A'])
def test_segment_eigenmid(mask, resolution, expected):
    """Test the 'SegmentEigenmid.compute' method."""

    # Assert the equality between actual and expected outcome
    assert isclose(
        SegmentEigenmid.compute(mask, resolution), expected, atol=1e-6)
