"""Machine learning model-based component retrieval."""

# Author: Tim Ortkamp

# %% Function definition


def get_machine_learning_components(components):
    """
    Get the machine learning model-based plan components.

    Parameters
    ----------
    components : list
        Plan components.

    Returns
    -------
    list
        Machine learning model-based plan components.
    """

    return list(
        component for component in components
        if component.get_class() == 'MachineLearningComponent')

def get_machine_learning_constraints(components):
    """
    Get the machine learning model-based plan constraints.

    Parameters
    ----------
    components : list
        Plan components.

    Returns
    -------
    list
        Machine learning model-based plan constraints.
    """

    return list(
        component for component in components
        if component.get_class() == 'MachineLearningComponent'
        and component.component_type == 'constraint')

def get_machine_learning_objectives(components):
    """
    Get the machine learning model-based plan objectives.

    Parameters
    ----------
    components : list
        Plan components.

    Returns
    -------
    list
        Machine learning model-based plan objectives.
    """

    return list(
        component for component in components
        if component.get_class() == 'MachineLearningComponent'
        and component.component_type == 'objective')
