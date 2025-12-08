"""Copycat function test."""

# Authors: Moritz Müller, Tim Ortkamp

# %% Internal package import

from pyanno4rt.base import (
    Configuration, Evaluation, Optimization, TreatmentPlan)
from pyanno4rt.optimization.components import (
    SquaredDeviation, SquaredOverdosing)
from pyanno4rt.tools import copycat

# %% Test definition


def test_copycat():
    """Test the 'copycat' function."""

    # Set the expected outcome
    expected = TreatmentPlan(

        configuration=Configuration(
            label='test',
            modality='photon',
            imaging_path='./docs/TG_119_data.mat',
            dose_matrix_path='./docs/TG_119_photonDij.mat',
            dose_resolution=[6, 6, 6],
            min_log_level='info',
            number_of_fractions=30),

        optimization=Optimization(
            components=[
                SquaredOverdosing(
                    segment='Core', maximum_dose=25,
                    component_type='objective', embedding='active', weight=100,
                    rank=1, bounds=None, identifier=None),
                SquaredDeviation(
                    segment='OuterTarget', target_dose=60,
                    component_type='objective', embedding='active',
                    weight=1000, rank=1, bounds=None, identifier=None),
                SquaredOverdosing(
                    segment='BODY', maximum_dose=30,
                    component_type='objective', embedding='active', weight=800,
                    rank=1, bounds=None, identifier=None)],
            method='weighted-sum',
            solver='scipy',
            algorithm='L-BFGS-B',
            initial_strategy='target-coverage',
            initial_fluence=None,
            lower_variable_bounds=0,
            upper_variable_bounds=None,
            maximum_iterations=500,
            tolerance=1e-3),

        evaluation=Evaluation(
            dvh_type='cumulative',
            number_of_points=1000,
            reference_volumes=[2, 5, 50, 95, 98],
            reference_doses=[])

        )

    # Get the actual outcome
    actual = copycat(TreatmentPlan, './tests/extra_files/copycat/test')

    # Assert the equality between actual and expected outcome
    assert actual.configuration.to_dict() == expected.configuration.to_dict()
    assert actual.optimization.to_dict() == expected.optimization.to_dict()
    assert actual.evaluation.to_dict() == expected.evaluation.to_dict()
