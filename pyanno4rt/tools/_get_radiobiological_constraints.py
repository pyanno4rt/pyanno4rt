"""Radiobiological constraint retrieval."""

# Author: Tim Ortkamp

# %% Internal package import

from pyanno4rt.tools import flatten

# %% Function definition


def get_radiobiological_constraints(segmentation):
    """
    Get a tuple with the set radiobiological model-based constraint functions.

    Parameters
    ----------
    segmentation : dict
        Dictionary with information on the segmented structures.

    Returns
    -------
    tuple
        Flattened tuple with the set radiobiological model-based constraint \
        functions.
    """

    return tuple(constraint for constraint in flatten(
        segmentation[segment]['constraint'] for segment in segmentation
        if segmentation[segment]['constraint'])
        if constraint.get_class() == 'RadiobiologicalComponent')
