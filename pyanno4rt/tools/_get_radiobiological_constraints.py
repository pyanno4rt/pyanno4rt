"""Radiobiological constraint retrieval."""

# Author: Tim Ortkamp

# %% Internal package import

from pyanno4rt.tools import flatten

# %% Function definition


def get_radiobiological_constraints(segmentation):
    """
    Get a tuple with the radiobiological model-based constraint functions.

    Parameters
    ----------
    segmentation : dict
        Dictionary with information on the segments.

    Returns
    -------
    tuple
        Tuple with the radiobiological model-based constraint functions.
    """

    return tuple(constraint for constraint in flatten(
        segmentation[segment]['constraint'] for segment in segmentation
        if segmentation[segment]['constraint'] is not None)
        if constraint.get_class() == 'RadiobiologicalComponent')
