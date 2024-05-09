"""Radiobiology objective retrieval."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% Internal package import

from pyanno4rt.tools import flatten

# %% Function definition


def get_radiobiology_objectives(segmentation):
    """
    Get a tuple with the set radiobiology model-based objective functions.

    Parameters
    ----------
    segmentation : dict
        Dictionary with information on the segmented structures.

    Returns
    -------
    tuple
        Flattened tuple with the set radiobiology model-based objective \
        functions.
    """

    return tuple(objective for objective in flatten(
        segmentation[segment]['objective'] for segment in segmentation
        if segmentation[segment]['objective'])
        if objective.RETURNS_OUTCOME and not objective.DEPENDS_ON_DATA)
