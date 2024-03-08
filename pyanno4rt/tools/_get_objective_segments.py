"""Objective segment retrieval."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% Internal package import

from pyanno4rt.tools import flatten

# %% Function definition


def get_objective_segments(segmentation):
    """
    Get a tuple with the segments associated with the objectives.

    Parameters
    ----------
    segmentation : dict
        Dictionary with information on the segmented structures.

    Returns
    -------
    tuple
        Flattened tuple with the segments associated with the objectives.
    """

    return tuple(flatten(
        [segment]*len(segmentation[segment]['objective'])
        if isinstance(segmentation[segment]['objective'], list)
        else [segment] for segment in segmentation
        if segmentation[segment]['objective']))
