"""Optimization and evaluation output resetting."""

# Author: Tim Ortkamp

# %% External package import

from numpy import empty

# %% Internal package import

from pyanno4rt.datahub import Datahub
from pyanno4rt.tools import (
    get_machine_learning_constraints, get_machine_learning_objectives)

# %% Function definition


def reset_outputs():
    """Reset the optimization and evaluation outputs."""

    # Initialize the datahub
    hub = Datahub()

    # Get the segmentation from the datahub
    segmentation = hub.segmentation

    # Check if lexicographic or weighted-sum optimization are selected
    if type(hub.optimization['problem']).__name__ in (
            'LexicographicOptimization', 'WeightedSumOptimization'):

        # Reset the optimization tracker
        hub.optimization['problem'].tracker = {
            key: [] for key in hub.optimization['problem'].tracker}

    # Loop over the machine learning model-based components
    for component in (
            get_machine_learning_constraints(segmentation)
            + get_machine_learning_objectives(segmentation)):

        # Get the feature calculator of the component
        feature_calculator = component.data_model_handler.feature_calculator

        # Reset the feature history
        feature_calculator.feature_history = empty(
            shape=(1, len(feature_calculator.feature_map)))
