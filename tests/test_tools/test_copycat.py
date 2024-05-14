"""Copycat function test."""

# Authors: Moritz Müller, Tim Ortkamp <tim.ortkamp@kit.edu>

# %% Internal package import

from pyanno4rt.base import TreatmentPlan
from pyanno4rt.tools import copycat

# %% Test definition


def test_copycat():
    """Test the 'copycat' function."""

    # Set the expected outcome
    expected = TreatmentPlan(

        configuration={
            'label': 'test',
            'min_log_level': 'info',
            'modality': 'photon',
            'number_of_fractions': 30,
            'imaging_path': './docs/TG_119_data.mat',
            'target_imaging_resolution': None,
            'dose_matrix_path': './docs/TG_119_photonDij.mat',
            'dose_resolution': [6, 6, 6]
            },

        optimization={
            'components': {
                'Core': {
                    'type': 'objective',
                    'instance': {
                        'class': 'Squared Overdosing',
                        'parameters': {
                            'maximum_dose': 25,
                            'weight': 100,
                            }
                        }
                    },
                'OuterTarget': {
                    'type': 'objective',
                    'instance': {
                        'class': 'Squared Deviation',
                        'parameters': {
                            'target_dose': 60,
                            'weight': 1000
                            }
                        }
                    },
                'BODY': {
                    'type': 'objective',
                    'instance': {
                        'class': 'Squared Overdosing',
                        'parameters': {
                            'maximum_dose': 30,
                            'weight': 800
                            }
                        }
                    }
                },
            'method': 'weighted-sum',
            'solver': 'scipy',
            'algorithm': 'L-BFGS-B',
            'initial_strategy': 'target-coverage',
            'initial_fluence_vector': None,
            'lower_variable_bounds': 0,
            'upper_variable_bounds': None,
            'max_iter': 500,
            'tolerance': 0.001
            },

        evaluation={
            'dvh_type': 'cumulative',
            'number_of_points': 1000,
            'reference_volume': [2, 5, 50, 95, 98],
            'reference_dose': [],
            'display_metrics': [],
            'display_segments': []
            }

        )

    # Get the actual outcome
    actual = copycat(TreatmentPlan, './tests/extra_files/copycat/test')

    # Assert the equality between actual and expected outcome
    assert actual.configuration == expected.configuration
    assert actual.optimization == expected.optimization
    assert actual.evaluation == expected.evaluation
