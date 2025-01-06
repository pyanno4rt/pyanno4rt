"""Segment volume feature test."""

# Authors: Moritz Müller, Tim Ortkamp

# %% External package import

from pytest import mark
from jax.numpy import array

# %% Internal package import

from pyanno4rt.learning_model.features import SegmentVolume

# %% Test definition


# Define the input test cases
case_A = array([[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]])
case_B = array([[[1, 0, 1], [1, 0, 1], [1, 0, 1]],
               [[1, 0, 1], [1, 0, 1], [1, 0, 1]],
               [[1, 0, 1], [1, 0, 1], [1, 0, 1]]])


# Define the argument sets
@mark.parametrize(
    'mask, spacing, expected',
    [(case_A, array([5, 5, 5]), 500), (case_B, array([1, 1, 1]), 18)],
    ids=['2D mask', '3D mask'])
def test_segment_volume(mask, spacing, expected):
    """Test the 'SegmentVolume.compute' method."""

    # Assert the equality between actual and expected outcome
    assert SegmentVolume.compute(mask, spacing) == expected
