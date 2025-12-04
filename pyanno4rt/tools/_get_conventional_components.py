"""Conventional component retrieval."""

# Author: Tim Ortkamp

# %% Function definition


def get_conventional_components(components):
    """
    Get the conventional plan components.

    Parameters
    ----------
    components : list
        Plan components.

    Returns
    -------
    list
        Conventional plan components.
    """

    return list(
        component for component in components
        if component.get_class() == 'ConventionalComponent')

def get_conventional_constraints(components):
    """
    Get the conventional plan constraints.

    Parameters
    ----------
    components : list
        Plan components.

    Returns
    -------
    list
        Conventional plan constraints.
    """

    return list(
        component for component in components
        if component.get_class() == 'ConventionalComponent'
        and component.component_type == 'constraint')

def get_conventional_objectives(components):
    """
    Get the conventional plan objectives.

    Parameters
    ----------
    components : list
        Plan components.

    Returns
    -------
    list
        Conventional plan objectives.
    """

    return list(
        component for component in components
        if component.get_class() == 'ConventionalComponent'
        and component.component_type == 'objective')
