"""Dose energy feature test."""

# Author: Moritz Müller, Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from numpy import ones, tile
from pytest import mark

# %% Internal package import

from pyanno4rt.learning_model.features.catalogue import DoseEnergy
from pyanno4rt.learning_model.features.catalogue._dose_energy import sigmoid

# %% Test definition


# Define the argument sets
@mark.parametrize(
    'value, multiplier, summand, expected',
    [(0, 0, 0, 0.5), ([0, 0], 10, 0, [0.5, 0.5])],
    ids=['single', 'multi'])
def test_sigmoid_energy(value, multiplier, summand, expected):
    """Test the 'sigmoid' function."""

    # Assert the equality between actual and expected outcome
    assert sigmoid(value, multiplier, summand) == expected


# Define the argument sets.
@mark.parametrize(
    'dose, expected',
    [(tile([0, 255], 128), 0.5), (ones(256), 256.0)],
    ids=['extrema', 'homogeneous'])
def test_dose_energy_function(dose, expected):
    """Test the 'DoseEnergy.function' method."""

    # Assert the equality between actual and expected outcome
    assert DoseEnergy.function(dose) == expected


# Define the argument sets.
@mark.parametrize(
    'dose, expected',
    [(tile([0, 255], 128), 0.5), (ones(256), 256.0)],
    ids=['extrema', 'homogeneous'])
def test_dose_energy_compute(dose, expected):
    """Test the 'DoseEnergy.compute' method."""

    # Assert the equality between actual and expected outcome
    assert DoseEnergy.compute(dose) == expected
