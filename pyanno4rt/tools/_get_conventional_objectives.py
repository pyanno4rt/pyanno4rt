"""Conventional objective retrieval."""

# Author: Tim Ortkamp

# %% Internal package import

from pyanno4rt.tools import flatten

# %% Function definition


def get_conventional_objectives(segmentation):
    """
    Get a tuple with all set conventional objective functions.

    Parameters
    ----------
    segmentation : dict
        Dictionary with information on the segmented structures.

    Returns
    -------
    tuple
        Tuple with all set conventional objective functions.
    """

    return tuple(objective for objective in flatten(
        segmentation[segment]['objective'] for segment in segmentation
        if segmentation[segment]['objective'] is not None)
        if objective.get_class() == 'ConventionalComponent')
