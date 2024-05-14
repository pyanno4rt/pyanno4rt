"""Machine learning model-based constraint retrieval."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% Internal package import

from pyanno4rt.tools import flatten

# %% Function definition


def get_machine_learning_constraints(segmentation):
    """
    Get a tuple with all set machine learning model-based constraint functions.

    Parameters
    ----------
    segmentation : dict
        Dictionary with information on the segmented structures.

    Returns
    -------
    tuple
        Flattened tuple with all set machine learning model-based constraint \
        functions.
    """

    return tuple(constraint for constraint in flatten(
        segmentation[segment]['constraint'] for segment in segmentation
        if segmentation[segment]['constraint'])
        if constraint.RETURNS_OUTCOME and constraint.DEPENDS_ON_DATA)
