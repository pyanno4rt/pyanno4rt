"""Dose-volume histogram abscissa feature test."""

# Authors: Moritz Müller, Tim Ortkamp

# %% External package import

from jax.numpy import array
from pytest import mark

# %% Internal package import

from pyanno4rt.learning.features import DoseDx

# %% Test definition


# Define the input test cases
case_A = array([1, 2, 3, 4]), 50
case_B = array([1, 2, 3, 4, 5]), 50
case_C = array([1, 2, 3]), 50


# Define the argument sets
@mark.parametrize(
    'dose, level, expected',
    [(*case_A, 3), (*case_B, 3), (*case_C, 3)],
    ids=['no round', 'round down', 'round up'])
def test_dose_dx_pyvalue(dose, level, expected):
    """Test the 'DoseDx.pyvalue' method."""

    # Assert the equality between actual and expected outcome
    assert DoseDx.pyvalue(dose, level) == expected


# Define the argument sets
@mark.parametrize(
    'dose, level, expected',
    [(*case_A, 3), (*case_B, 4), (*case_C, 3)],
    ids=['no round', 'round up from 2.5', 'round up from 1.5'])
def test_dose_dx_matvalue(dose, level, expected):
    """Test the 'DoseDx.matvalue' method."""

    # Assert the equality between actual and expected outcome
    assert DoseDx.matvalue(dose, level) == expected


# Define the argument sets
@mark.parametrize(
    'dose, level, expected',
    [(*case_A, 3), (*case_B, 4), (*case_C, 3)],
    ids=['no round', 'round up from 2.5', 'round up from 1.5'])
def test_dose_dx_compute(dose, level, expected):
    """Test the static method 'DoseDx.compute'."""

    # Assert the equality between actual and expected outcome
    assert DoseDx.compute(dose, level) == expected
