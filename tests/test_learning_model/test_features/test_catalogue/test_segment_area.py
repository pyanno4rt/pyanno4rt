"""Segment area feature test."""

# Author: Moritz Müller, Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from jax.numpy import array, identity, zeros
from pytest import mark

# %% Internal package import

from pyanno4rt.learning_model.features.catalogue import SegmentArea

# %% Test definition


# Define the input test cases
case_A = (array([identity(3), identity(3), identity(3)]), (1, 1, 1))
case_B = (zeros((3, 3, 3)).at[(1, 1, 0), (1, 1, 1), (1, 2, 1)].set(1),
          (2, 2, 2))


# Define the argument sets
@mark.parametrize(
    'mask, spacing, expected',
    [(*case_A, 21), (*case_B, 28)],
    ids=['case_A', 'case_B'])
def test_segment_area(mask, spacing, expected):
    """Test the 'SegmentArea.compute' method."""

    # Assert the equality between actual and expected outcome
    assert SegmentArea.compute(mask, spacing) == expected
