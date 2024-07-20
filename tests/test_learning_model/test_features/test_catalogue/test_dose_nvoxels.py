"""Dose voxel number feature test."""

# Author: Moritz Müller, Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from jax.numpy import array, array_equal
from pytest import mark

# %% Internal package import

from pyanno4rt.learning_model.features.catalogue import DoseNVoxels

# %% Test definition


# Define the argument sets
@mark.parametrize(
    'dose, expected',
    [((1, 2, 3, 4), 4), ((), 0)],
    ids=['case A', 'case B'])
def test_dose_nvoxels_function(dose, expected):
    """Test the 'DoseNVoxels.function' method."""

    # Assert the equality between actual and expected outcome
    assert DoseNVoxels.function(dose) == expected


# Define the argument sets
@mark.parametrize(
    'dose, expected',
    [((1, 2, 3, 4), 4), ((), 0)],
    ids=['case A', 'case B'])
def test_dose_nvoxels_compute(dose, expected):
    """Test the 'DoseNVoxels.compute' method."""

    # Assert the equality between actual and expected outcome
    assert DoseNVoxels.compute(dose) == expected


# Define the argument sets
@mark.parametrize(
    'dose, dimension, indices, expected',
    [((1.0, 2.0, 3.0, 4.0), 4, (0, 1, 2, 3), array([0, 0, 0, 0])),
     ((0.0, 1.0, 0.0), 3, (0, 1, 2), array([0, 0, 0]))],
    ids=['case A', 'case B'])
def test_dose_nvoxels_differentiate(dose, dimension, indices, expected):
    """Test the 'DoseNVoxels.differentiate' method."""

    # Assert the equality between actual and expected outcome
    assert array_equal(
        DoseNVoxels.differentiate(dose, dimension, indices).toarray()[0],
        expected)
