"""Radiobiology constraint retrieval."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% Internal package import

from pyanno4rt.tools import flatten

# %% Function definition


def get_radiobiology_constraints(segmentation):
    """
    Get a tuple with the set radiobiology model-based constraint functions.

    Parameters
    ----------
    segmentation : dict
        Dictionary with information on the segmented structures.

    Returns
    -------
    tuple
        Flattened tuple with the set radiobiology model-based constraint \
        functions.
    """

    return tuple(constraint for constraint in flatten(
        segmentation[segment]['constraint'] for segment in segmentation
        if segmentation[segment]['constraint'])
        if constraint.RETURNS_OUTCOME and not constraint.DEPENDS_ON_DATA)
