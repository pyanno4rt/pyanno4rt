"""Constraints return."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% Internal package import

from pyanno4rt.tools import flatten

# %% Function definition


def get_all_constraints(segmentation):
    """
    Return a tuple with the user-assigned constraints.

    Parameters
    ----------
    segmentation : dict
        Dictionary with information on the segmented structures.

    Returns
    -------
    tuple
        Flattened tuple with the user-assigned constraints.
    """

    return tuple(constraint for constraint in flatten(
        segmentation[segment]['constraint'] for segment in segmentation
        if segmentation[segment]['constraint']))
