"""Machine learning objectives return."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% Internal package import

from pyanno4rt.tools import flatten

# %% Function definition


def get_machine_learning_objectives(segmentation):
    """
    Return a tuple with the user-assigned, machine learning model-based \
    objectives.

    Parameters
    ----------
    segmentation : dict
        Dictionary with information on the segmented structures.

    Returns
    -------
    tuple
        Flattened tuple with the user-assigned, machine learning model-based \
        objectives.
    """

    return tuple(objective for objective in flatten(
        segmentation[segment]['objective'] for segment in segmentation
        if segmentation[segment]['objective'])
        if objective.RETURNS_OUTCOME and objective.DEPENDS_ON_DATA)
