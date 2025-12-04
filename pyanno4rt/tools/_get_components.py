"""Component retrieval."""

# Author: Tim Ortkamp

# %% Function definition


def get_constraints(components):
    """
    Return the plan constraints.

    Parameters
    ----------
    components : list
        Plan components.

    Returns
    -------
    list
        Plan constraints.
    """

    return list(
        component for component in components
        if component.component_type == 'constraint')

def get_objectives(components):
    """
    Return the plan objectives.

    Parameters
    ----------
    components : list
        Plan components.

    Returns
    -------
    list
        Plan objectives.
    """

    return list(
        component for component in components
        if component.component_type == 'objective')
