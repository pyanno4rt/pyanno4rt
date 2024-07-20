"""Dose minimum feature test."""

# Authors: Moritz Müller, Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from jax.numpy import array, array_equal
from pytest import mark

# %% Internal package import

from pyanno4rt.learning_model.features.catalogue import DoseMinimum

# %% Test definition


# Define the input test cases
case_A = array([1.0, 2.0, 3.0, 4.0])
case_B = array([1.0, 1.0, 2.0, 3.0, 4.0])


# Define the argument sets
@mark.parametrize(
    'dose, expected',
    [(case_A, 1.0), (case_B, 1.0)],
    ids=['unique', 'non-unique'])
def test_dose_minimum_function(dose, expected):
    """Test the 'DoseMinimum.function' method."""

    # Assert the equality between actual and expected outcome
    assert DoseMinimum.function(dose) == expected


# Define the argument sets
@mark.parametrize(
    'dose, expected',
    [(case_A, 1.0), (case_B, 1.0)],
    ids=['unique', 'non-unique'])
def test_dose_minimum_compute(dose, expected):
    """Test the 'DoseMinimum.compute' method."""

    # Assert the equality between actual and expected outcome
    assert DoseMinimum.compute(dose) == expected


# Define the argument sets
@mark.parametrize(
    'dose, dimension, indices, expected',
    [(case_A, 4, (0, 1, 2, 3), array([1.0, 0.0, 0.0, 0.0])),
     (case_B, 5, (0, 1, 2, 3, 4), array([0.5, 0.5, 0.0, 0.0, 0.0]))],
    ids=['unique', 'non-unique'])
def test_dose_minimum_differentiate(dose, dimension, indices, expected):
    """Test the 'DoseMinimum.differentiate' method."""

    # Assert the equality between actual and expected outcome
    assert array_equal(
        DoseMinimum.differentiate(dose, dimension, indices).toarray()[0],
        expected)
