"""Dose skewness feature test."""

# Author: Moritz Müller, Tim Ortkamp <tim.ortkamp.kit.edu>

# %% External package import

from jax.numpy import allclose, array, isclose
from pytest import mark

# %% Internal package import

from pyanno4rt.learning_model.features.catalogue import DoseSkewness

# %% Test definition


# Define the input test cases
case_A = array([2, 4, 7, 1, 3, 5, 9, 6, 8, 10])
case_B = array([1, 2, 2, 3, 4, 5, 6, 7, 8, 10])


# Define the argument sets
@mark.parametrize(
    'dose, expected',
    [(case_A, 0), (case_B, 0.37304196)],
    ids=['case A', 'case B'])
def test_dose_skewness_function(dose, expected):
    """Test the 'DoseSkewness.function' method."""

    # Asser the equality between actual and expected outcome
    assert isclose(DoseSkewness.function(dose), expected)


# Define the argument sets
@mark.parametrize(
    'dose, expected',
    [(case_A, 0), (case_B, 0.37304196)],
    ids=['case A', 'case B'])
def test_dose_skewness_compute(dose, expected):
    """Test the 'DoseSkewness.compute' method."""

    # Asser the equality between actual and expected outcome
    assert isclose(DoseSkewness.compute(dose), expected)


# Define the argument sets
@mark.parametrize(
    'dose, dimension, indices, expected',
    [(array([1.0, 2.0, 3.0, 4.0, 5.0]), 5, (0, 1, 2, 3, 4),
      array([0.42426407, -0.21213203, -0.42426407, -0.21213203, 0.42426407]))],
    ids=['case A'])
def test_dose_skewness_differentiate(dose, dimension, indices, expected):
    """Test the 'DoseSkewness.differentiate' method."""

    # Assert the equality between actual and expected outcome
    assert allclose(
        DoseSkewness.differentiate(dose, dimension, indices).toarray()[0],
        expected)
