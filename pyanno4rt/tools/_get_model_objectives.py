"""Model objectives return."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% Internal package import

from pyanno4rt.tools import flatten

# %% Function definition


def get_model_objectives(segmentation):
    """
    Return a tuple with the user-assigned, model-dependent objectives.

    Parameters
    ----------
    segmentation : dict
        Dictionary with information on the segmented structures.

    Returns
    -------
    tuple
        Flattened tuple with the user-assigned, model-dependent objectives.
    """

    return tuple(objective
                 for objective in flatten(
                         segmentation[segment]['objective']
                         for segment in segmentation
                         if segmentation[segment]['objective'])
                 if objective.DEPENDS_ON_MODEL)
