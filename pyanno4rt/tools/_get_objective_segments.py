"""Objective segments return."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% Internal package import

from pyanno4rt.tools import flatten

# %% Function definition


def get_objective_segments(segmentation):
    """
    Return a tuple with the segments associated with each objective.

    Parameters
    ----------
    segmentation : dict
        Dictionary with information on the segmented structures.

    Returns
    -------
    tuple
        Flattened tuple with the segments associated with each objective.
    """

    return tuple(flatten([segment]*len(segmentation[segment]['objective'])
                         if isinstance(segmentation[segment]['objective'],
                                       (tuple, list))
                         else [segment] for segment in segmentation
                         if segmentation[segment]['objective']))
