"""Constraint segment retrieval."""

# Author: Tim Ortkamp

# %% Internal package import

from pyanno4rt.tools import flatten

# %% Function definition


def get_constraint_segments(segmentation):
    """
    Get a tuple with the segments associated with the constraints.

    Parameters
    ----------
    segmentation : dict
        Dictionary with information on the segments.

    Returns
    -------
    tuple
        Tuple with the segments associated with the constraints.
    """

    return tuple(flatten(
        [segment]*len(segmentation[segment]['constraint'])
        if isinstance(segmentation[segment]['constraint'], list)
        else [segment] for segment in segmentation
        if segmentation[segment]['constraint'] is not None))
