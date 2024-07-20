"""Dose-volume histogram ordinate feature test."""

# Author: Moritz Müller, Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from jax.numpy import array, array_equal
from pytest import mark

# %% Internal package import

from pyanno4rt.learning_model.features.catalogue import DoseVx

# %% Test definition


# Define the input test cases
case_A = (1, array([1, 2, 3]), 1)
case_B = (3, array([1, 2, 3, 4]), 0.5)


# Define the argument sets
@mark.parametrize(
    'level, dose, expected', [case_A, case_B], ids=['case A', 'case B'])
def test_dose_vx_function(level, dose, expected):
    """Test the 'DoseVx.function' method."""

    # Assert the equality between actual and expected outcome
    assert DoseVx.function(level, dose) == expected


# Define the argument sets
@mark.parametrize(
    'level, dose, expected', [case_A, case_B], ids=['case A', 'case B'])
def test_dose_vx_compute(level, dose, expected):
    """Test the 'DoseVx.compute' method."""

    # Assert the equality between actual and expected outcome
    assert DoseVx.compute(level, dose) == expected


# Define the argument sets
@mark.parametrize(
    'level, dose, dimension, indices, expected',
    [(1.0, array([1.0, 2.0, 5.0]), 3, (0, 1, 2), array([0, 0, 0])),
     (2.0, array([1.0, 2.0, 3.0]), 3, (0, 1, 2), array([0, 0, 0]))],
    ids=['case A', 'case B'])
def test_dose_vx_differentiate(level, dose, dimension, indices, expected):
    """Test the 'DoseVx.differentiate' method."""

    # Assert the equality between actual and expected outcome
    assert array_equal(
        DoseVx.differentiate(level, dose, dimension, indices).toarray()[0],
        expected)
