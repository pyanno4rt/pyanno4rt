"""Dose mean feature test."""

# Author: Moritz Müller, Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from jax.numpy import array, array_equal
from pytest import mark

# %% Internal package import

from pyanno4rt.learning_model.features.catalogue import DoseMean

# %% Test definition


# Define the input test cases
case_A = array([1.0, 2.0, 3.0, 4.0])
case_B = array([2, 4, 6, 8])


# Define the argument sets
@mark.parametrize(
    'dose, expected',
    [(case_A, 2.5), (case_B, 5)],
    ids=['case A', 'case B'])
def test_dose_mean_function(dose, expected):
    """Test the 'DoseMean.function' method."""

    # Assert the equality between actual and expected outcome
    assert DoseMean.function(dose) == expected


# Define the argument sets
@mark.parametrize(
    'dose, expected',
    [(case_A, 2.5), (case_B, 5)],
    ids=['case A', 'case B'])
def test_dose_mean_compute(dose, expected):
    """Test the 'DoseMean.compute' method."""

    # Assert the equality between actual and expected outcome
    assert DoseMean.compute(dose) == expected


# Define the argument sets
@mark.parametrize(
    'dose, dimension, indices, expected',
    [(case_A, 4, (0, 1, 2, 3), array([0.25, 0.25, 0.25, 0.25])),
     (array([7.5, 3.3]), 2, (0, 1), array([0.5, 0.5]))],
    ids=['case A', 'case B'])
def test_dose_mean_differentiate(dose, dimension, indices, expected):
    """Test the 'DoseMean.differentiate' method."""

    # Assert the equality between actual and expected outcome
    assert array_equal(
        DoseMean.differentiate(dose, dimension, indices).toarray()[0],
        expected)
