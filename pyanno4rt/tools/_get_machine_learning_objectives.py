"""Machine learning model-based objective retrieval."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% Internal package import

from pyanno4rt.tools import flatten

# %% Function definition


def get_machine_learning_objectives(segmentation):
    """
    Get a tuple with all set machine learning model-based objective functions.

    Parameters
    ----------
    segmentation : dict
        Dictionary with information on the segmented structures.

    Returns
    -------
    tuple
        Flattened tuple with all set machine learning model-based objective \
        functions.
    """

    return tuple(objective for objective in flatten(
        segmentation[segment]['objective'] for segment in segmentation
        if segmentation[segment]['objective'])
        if objective.RETURNS_OUTCOME and objective.DEPENDS_ON_DATA)
