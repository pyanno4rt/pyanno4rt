"""Conventional objectives return."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% Internal package import

from pyanno4rt.tools import flatten

# %% Function definition


def get_conventional_objectives(segmentation):
    """
    Return a tuple with the user-assigned conventional objectives.

    Parameters
    ----------
    segmentation : dict
        Dictionary with information on the segmented structures.

    Returns
    -------
    tuple
        Flattened tuple with the user-assigned conventional objectives.
    """

    return tuple(objective
                 for objective in flatten(
                         segmentation[segment]['objective']
                         for segment in segmentation
                         if segmentation[segment]['objective'])
                 if not objective.RETURNS_OUTCOME
                 and not objective.DEPENDS_ON_DATA)
