"""Dose-volume histogram ordinate feature test."""

# Authors: Moritz Müller, Tim Ortkamp

# %% External package import

from jax.numpy import array, array_equal
from pytest import mark

# %% Internal package import

from pyanno4rt.learning.features import DoseVx

# %% Test definition


# Define the input test cases
case_A = (array([1, 2, 3]), 1, 1)
case_B = (array([1, 2, 3, 4]), 3, 0.5)


# Define the argument sets
@mark.parametrize(
    'dose, level, expected', [case_A, case_B], ids=['case A', 'case B'])
def test_dose_vx_value(dose, level, expected):
    """Test the 'DoseVx.value' method."""

    # Assert the equality between actual and expected outcome
    assert DoseVx.value(dose, level) == expected


# Define the argument sets
@mark.parametrize(
    'dose, level, expected', [case_A, case_B], ids=['case A', 'case B'])
def test_dose_vx_compute(dose, level, expected):
    """Test the 'DoseVx.compute' method."""

    # Assert the equality between actual and expected outcome
    assert DoseVx.compute(dose, level) == expected


# Define the argument sets
@mark.parametrize(
    'dose, level, dimension, indices, expected',
    [(array([1.0, 2.0, 5.0]), 1.0, 3, (0, 1, 2), array([0, 0, 0])),
     (array([1.0, 2.0, 3.0]), 2.0, 3, (0, 1, 2), array([0, 0, 0]))],
    ids=['case A', 'case B'])
def test_dose_vx_differentiate(dose, level, dimension, indices, expected):
    """Test the 'DoseVx.differentiate' method."""

    # Assert the equality between actual and expected outcome
    assert array_equal(
        DoseVx.differentiate(dose, level, dimension, indices), expected)
