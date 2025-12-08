"""Dose energy feature test."""

# Authors: Moritz Müller, Tim Ortkamp

# %% External package import

from numpy import ones, tile
from pytest import mark

# %% Internal package import

from pyanno4rt.learning.features import DoseEnergy
from pyanno4rt.learning.features._dose_energy import sigmoid

# %% Test definition


# Define the argument sets
@mark.parametrize(
    'value, expected',
    [(0, 0.5)],
    ids=['single'])
def test_sigmoid_energy(value, expected):
    """Test the 'sigmoid' function."""

    # Assert the equality between actual and expected outcome
    assert sigmoid(value) == expected


# Define the argument sets.
@mark.parametrize(
    'dose, expected',
    [(tile([0, 255], 128), 0.5), (ones(256), 256.0)],
    ids=['extrema', 'homogeneous'])
def test_dose_energy_value(dose, expected):
    """Test the 'DoseEnergy.value' method."""

    # Assert the equality between actual and expected outcome
    assert DoseEnergy.value(dose) == expected


# Define the argument sets.
@mark.parametrize(
    'dose, expected',
    [(tile([0, 255], 128), 0.5), (ones(256), 256.0)],
    ids=['extrema', 'homogeneous'])
def test_dose_energy_compute(dose, expected):
    """Test the 'DoseEnergy.compute' method."""

    # Assert the equality between actual and expected outcome
    assert DoseEnergy.compute(dose) == expected
