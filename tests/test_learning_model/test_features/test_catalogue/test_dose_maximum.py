"""Dose maximum feature test."""

# Authors: Moritz Müller, Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from jax.numpy import array, array_equal
from pytest import mark

# %% Internal package import

from pyanno4rt.learning_model.features.catalogue import DoseMaximum

# %% Test definition


# Define the input test cases
case_A = array([1, 2, 4, 4])
case_B = array([1, 2, 3, 4.0, 4])


# Define the argument sets
@mark.parametrize(
    'dose, expected',
    [(case_A, 4), (case_B, 4)],
    ids=['integer', 'double'])
def test_dose_maximum_function(dose, expected):
    """Test the 'DoseMaximum.function' method."""

    # Assert the equality between actual and expected outcome
    assert DoseMaximum.function(dose) == expected


# Define the argument sets
@mark.parametrize(
    'dose, expected',
    [(case_A, 4), (case_B, 4)],
    ids=['integer', 'double'])
def test_dose_maximum_compute(dose, expected):
    """Test the 'DoseMaximum.compute' method."""

    # Assert the equality between actual and expected outcome
    assert DoseMaximum.compute(dose) == expected


# Define the argument sets
@mark.parametrize(
    'dose, dimension, indices, expected',
    [(array([1.0, 2.0, 3.0, 4.0]), 4, (0, 1, 2, 3),
      array([0.0, 0.0, 0.0, 1.0])),
     (array([1.0, 2.0, 4.0, 4.0]), 4, (0, 1, 2, 3),
      array([0.0, 0.0, 0.5, 0.5]))],
    ids=['unique', 'non-unique'])
def test_dose_maximum_differentiate(dose, dimension, indices, expected):
    """Test the 'DoseMaximum.differentiate' method."""

    # Assert the equality between actual and expected outcome
    assert array_equal(
        DoseMaximum.differentiate(dose, dimension, indices).toarray()[0],
        expected)
