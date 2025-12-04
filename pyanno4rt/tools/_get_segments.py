"""Component segment retrieval."""

# Author: Tim Ortkamp

# %% Internal package import

from pyanno4rt.tools import flatten

# %% Function definition


def get_all_segments(components):
    """
    Get the segments associated with the plan components.

    Parameters
    ----------
    components : list
        Plan components.

    Returns
    -------
    list
        Segments associated with the plan components.
    """

    return list(set(flatten(component.segment for component in components)))

def get_constraint_segments(components):
    """
    Get the segments associated with the plan constraints.

    Parameters
    ----------
    components : list
        Plan components.

    Returns
    -------
    list
        Segments associated with the plan constraints.
    """

    return list(set(flatten(
        component.segment for component in components
        if component.component_type == 'constraint')))

def get_objective_segments(components):
    """
    Get the segments associated with the plan objectives.

    Parameters
    ----------
    components : list
        Plan components.

    Returns
    -------
    list
        Segments associated with the plan objectives.
    """

    return list(set(flatten(
        component.segment for component in components
        if component.component_type == 'objective')))
