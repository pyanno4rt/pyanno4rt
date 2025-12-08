"""Dose entropy feature test."""

# Authors: Moritz Müller, Tim Ortkamp

# %% External package import

from numpy import ones, tile
from pytest import mark

# %% Internal package import

from pyanno4rt.learning.features import DoseEntropy
from pyanno4rt.learning.features._dose_entropy import sigmoid

# %% Test definition


# Define the argument sets
@mark.parametrize(
    'value, expected',
    [(0, 0.5)],
    ids=['single'])
def test_sigmoid_entropy(value, expected):
    """Test the 'sigmoid' function."""

    # Assert the equality between actual and expected outcome
    assert sigmoid(value) == expected


# Define the argument sets
@mark.parametrize(
    'dose, expected',
    [(tile([0, 255], 128), 1.0), (ones(256), 0.0)],
    ids=['extrema', 'homogeneous'])
def test_dose_entropy_value(dose, expected):
    """Test the 'DoseEntropy.value' method."""

    # Assert the equality between actual and expected outcome
    assert DoseEntropy.value(dose) == expected


# Define the argument sets
@mark.parametrize(
    'dose, expected',
    [(tile([0, 255], 128), 1.0), (ones(256), 0.0)],
    ids=['extrema', 'homogeneous'])
def test_dose_entropy_compute(dose, expected):
    """Test the 'DoseEntropy.compute' method."""

    # Assert the equality between actual and expected outcome
    assert DoseEntropy.compute(dose) == expected
