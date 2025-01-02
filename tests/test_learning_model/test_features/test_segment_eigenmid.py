"""Segment middle eigenvalue feature test."""

# Author: Moritz Müller, Tim Ortkamp <tim.ortkamp@it.edu>

# %% External package import

from jax.numpy import array, identity, isclose
from pytest import mark

# %% Internal package import

from pyanno4rt.learning_model.features import SegmentEigenmid

# %% Test definition


# Define the argument sets
@mark.parametrize(
    'mask, spacing, expected',
    [(*(array([identity(3), identity(3), identity(3)]), (1, 1, 1)), 6)],
    ids=['case_A'])
def test_segment_eigenmid(mask, spacing, expected):
    """Test the 'SegmentEigenmid.compute' method."""

    # Assert the equality between actual and expected outcome
    assert isclose(SegmentEigenmid.compute(mask, spacing), expected, atol=1e-6)
