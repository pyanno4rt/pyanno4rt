"""Dose deviation feature test."""

# Author: Moritz Müller, Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from jax.numpy import allclose, array, sqrt
from pytest import mark

# %% Internal package import

from pyanno4rt.learning_model.features.catalogue import DoseDeviation

# %% Test definition


# Define the input test cases
case_A = array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
case_B = array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0])


# Define the argument sets
@mark.parametrize(
    'dose, expected',
    [(case_A, 2.0), (case_B, 4.0)],
    ids=['case_A', 'case_B'])
def test_dose_deviation_function(dose, expected):
    """Test the 'DoseDeviation.function' method."""

    # Assert the equality between actual and expected outcome
    assert DoseDeviation.function(dose) == expected


# Define the argument sets
@mark.parametrize(
    'dose, expected',
    [(case_A, 2.0), (case_B, 4.0)],
    ids=['case_A', 'case_B'])
def test_dose_deviation_compute(dose, expected):
    """Test the 'DoseDeviation.compute' method."""

    # Assert the equality between actual and expected outcome
    assert DoseDeviation.compute(dose) == expected


# Define the argument sets
@mark.parametrize(
    'dose, dimension, indices, expected',
    [(array([2.0, 4.0, 6.0, 8.0]), 4, (0, 1, 2, 3),
      array([value/(4*sqrt(5)) for value in (-3, -1, 1, 3)])),
     (array([1.0, 3.0, 5.0, 7.0, 9.0]), 5, (0, 1, 2, 3, 4),
      array([value/(5*sqrt(2)) for value in (-2, -1, 0, 1, 2)]))],
    ids=['case_A', 'case_B'])
def test_dose_deviation_differentiate(dose, dimension, indices, expected):
    """Test the 'DoseDeviation.differentiate' method."""

    # Assert the equality between actual and expected outcome
    assert allclose(
        DoseDeviation.differentiate(dose, dimension, indices).toarray(),
        expected)
