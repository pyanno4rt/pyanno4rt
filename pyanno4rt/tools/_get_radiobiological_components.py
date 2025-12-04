"""Radiobiological component retrieval."""

# Author: Tim Ortkamp

# %% Function definition


def get_radiobiological_components(components):
    """
    Get the radiobiological model-based plan components.

    Parameters
    ----------
    components : list
        Plan components.

    Returns
    -------
    list
        Radiobiological model-based plan components.
    """

    return list(
        component for component in components
        if component.get_class() == 'RadiobiologicalComponent')

def get_radiobiological_constraints(components):
    """
    Get the radiobiological model-based plan constraints.

    Parameters
    ----------
    components : list
        Plan components.

    Returns
    -------
    list
        Radiobiological model-based plan constraints.
    """

    return list(
        component for component in components
        if component.get_class() == 'RadiobiologicalComponent'
        and component.component_type == 'constraint')

def get_radiobiological_objectives(components):
    """
    Get the radiobiological model-based plan objectives.

    Parameters
    ----------
    components : list
        Plan components.

    Returns
    -------
    list
        Radiobiological model-based plan objectives.
    """

    return list(
        component for component in components
        if component.get_class() == 'RadiobiologicalComponent'
        and component.component_type == 'objective')
