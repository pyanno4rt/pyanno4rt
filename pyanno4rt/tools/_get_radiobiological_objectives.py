"""Radiobiological objective retrieval."""

# Author: Tim Ortkamp

# %% Internal package import

from pyanno4rt.tools import flatten

# %% Function definition


def get_radiobiological_objectives(segmentation):
    """
    Get a tuple with the set radiobiological model-based objective functions.

    Parameters
    ----------
    segmentation : dict
        Dictionary with information on the segmented structures.

    Returns
    -------
    tuple
        Flattened tuple with the set radiobiological model-based objective \
        functions.
    """

    return tuple(objective for objective in flatten(
        segmentation[segment]['objective'] for segment in segmentation
        if segmentation[segment]['objective'])
        if objective.get_class() == 'RadiobiologicalComponent')
