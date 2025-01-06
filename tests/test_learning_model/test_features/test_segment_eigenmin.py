"""Segment minimum eigenvalue feature test."""

# Authors: Moritz Müller, Tim Ortkamp

# %% External package import

from jax.numpy import array, identity, isclose
from pytest import mark

# %% Internal package import

from pyanno4rt.learning_model.features import SegmentEigenmin

# %% Test definition


# Define the argument sets
@mark.parametrize(
    'mask, spacing, expected',
    [(*(array([identity(3), identity(3), identity(3)]), (1, 1, 1)), 0)],
    ids=['case_A'])
def test_segment_eigenmin(mask, spacing, expected):
    """Test the 'SegmentEigenmin.compute' method."""

    # Assert the equality between actual and expected outcome
    assert isclose(SegmentEigenmin.compute(mask, spacing), expected, atol=1e-6)
