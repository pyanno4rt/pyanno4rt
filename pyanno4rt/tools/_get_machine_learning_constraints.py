"""Machine learning model-based constraint retrieval."""

# Author: Tim Ortkamp

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
        Tuple with all set machine learning model-based constraint functions.
    """

    return tuple(constraint for constraint in flatten(
        segmentation[segment]['constraint'] for segment in segmentation
        if segmentation[segment]['constraint'] is not None)
        if constraint.get_class() == 'MachineLearningComponent')
