"""Dose kurtosis feature test."""

# Author: Moritz Müller, Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from jax.numpy import allclose, array, isclose
from pytest import mark

# %% Internal package import

from pyanno4rt.learning_model.features.catalogue import DoseKurtosis

# %% Test definition


# Define the argument sets
@mark.parametrize(
    'dose, expected',
    [(array([1, 2, 3, 4, 5]), 1.7), (array([-2, -1, 0, 1, 2]), 1.7)],
    ids=['case_A', 'case_B'])
def test_dose_kurtosis_function(dose, expected):
    """Test the 'DoseKurtosis.function' method."""

    # Assert the equality between actual and expected outcome
    assert isclose(DoseKurtosis.function(dose), expected)


# Define the argument sets
@mark.parametrize(
    'dose, expected',
    [(array([1, 2, 3, 4, 5]), 1.7), (array([-2, -1, 0, 1, 2]), 1.7)],
    ids=['case A', 'case B'])
def test_dose_kurtosis_compute(dose, expected):
    """Test the 'DoseKurtosis.compute' method."""

    # Assert the equality between actual and expected outcome
    assert isclose(DoseKurtosis.compute(dose), expected)


# Define the input test cases
case_A = (array([1.0, 2.0, 3.0, 4.0, 5.0]), 5, (0, 1, 2, 3, 4),
          array([-0.24, 0.48, 0, -0.48, 0.24]))
case_B = (array([2.0, 4.0, 6.0, 8.0, 10.0]), 5, (0, 1, 2, 3, 4),
          array([-0.12, 0.24, 0, -0.24, 0.12]))


# Define the argument sets
@mark.parametrize(
    'dose, dimension, indices, expected',
    [case_A, case_B],
    ids=['case_A', 'case_B'])
def test_dose_kurtosis_differentiate(dose, dimension, indices, expected):
    """Test the 'DoseKurtosis.differentiate' method."""

    # Assert the equality between actual and expected outcome
    assert allclose(
        DoseKurtosis.differentiate(dose, dimension, indices).toarray()[0],
        expected)
