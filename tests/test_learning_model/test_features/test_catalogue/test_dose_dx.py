"""Dose-volume histogram abscissa feature test."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from jax.numpy import array
from pytest import mark

# %% Internal package import

from pyanno4rt.learning_model.features.catalogue import DoseDx

# %% Test definition


# Define the input test cases
case_A = 50, array([1, 2, 3, 4])
case_B = 50, array([1, 2, 3, 4, 5])
case_C = 50, array([1, 2, 3])


# Define the argument sets
@mark.parametrize(
    'level, dose, expected',
    [(*case_A, 3), (*case_B, 3), (*case_C, 3)],
    ids=['no round', 'round down', 'round up'])
def test_dose_dx_pyfunction(level, dose, expected):
    """Test the 'DoseDx.pyfunction' method."""

    # Assert the equality between actual and expected outcome
    assert DoseDx.pyfunction(level, dose) == expected


# Define the argument sets
@mark.parametrize(
    'level, dose, expected',
    [(*case_A, 3), (*case_B, 4), (*case_C, 3)],
    ids=['no round', 'round up from 2.5', 'round up from 1.5'])
def test_dose_dx_matfunction(level, dose, expected):
    """Test the 'DoseDx.matfunction' method."""

    # Assert the equality between actual and expected outcome
    assert DoseDx.matfunction(level, dose) == expected


# Define the argument sets
@mark.parametrize(
    'level, dose, expected',
    [(*case_A, 3), (*case_B, 4), (*case_C, 3)],
    ids=['no round', 'round up from 2.5', 'round up from 1.5'])
def test_dose_dx_compute(level, dose, expected):
    """Test the static method 'DoseDx.compute'."""

    # Assert the equality between actual and expected outcome
    assert DoseDx.compute(level, dose) == expected
