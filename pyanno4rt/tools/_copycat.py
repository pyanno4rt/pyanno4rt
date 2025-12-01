"""Instance copycat."""

# Author: Tim Ortkamp

# %% External package import

from os import listdir, walk

from json import load
from numpy import load as npload

# %% Internal package import

from pyanno4rt.tools import apply

# %% Function definition


def copycat(base_class, path, ignore_optimum=False):
    """
    Create a copycat from a treatment plan snapshot.

    Parameters
    ----------
    base_class : :class:`~pyanno4rt.base._treatment_plan.TreatmentPlan`
        The base class from which to create an instance.

    path : str
        Directory path of the snapshot.

    ignore_optimum : bool
        Indicator for ignoring the optimal fluence file (if available).

    Returns
    -------
    object of class :class:`~pyanno4rt.base._treatment_plan.TreatmentPlan`
        The object used to represent the treatment plan.
    """

    def update_paths(inputs):
        """Update the model folder and data path for a component."""

        # Get the model path and the component
        path, component = inputs

        # Update the model path
        component.model_parameters.model_folder_path = path

        # Reset the data path
        component.model_parameters.data_path = None

        # Loop over the model path files
        for filename in listdir(path):

            # Check if the data file exists
            if 'model_data' in filename:

                # Update the data path
                component.model_parameters.data_path = f'{path}/{filename}'

    # Open a file stream
    with open(f'{path}/input_parameters.json', 'r', encoding='utf-8') as file:

        # Load the input parameter dictionaries
        input_parameters = load(file)

    # Loop over the path files
    for filename in listdir(path):

        # Check if the current file holds the patient data
        if 'patient_data' in filename:

            # Update the imaging path
            input_parameters['configuration']['imaging_path'] = (
                f'{path}/{filename}')

        # Check if the current file holds the dose influence matrix
        elif 'dose_influence_matrix' in filename:

            # Update the dose path
            input_parameters['configuration']['dose_matrix_path'] = (
                f'{path}/{filename}')

    # Initialize the treatment plan instance
    treatment_plan = base_class(**input_parameters)

    # Get the folder/model links
    links = (
        (f'{path}/{folder_name}', next(
            component for component in treatment_plan.optimization.components
            if (hasattr(component, 'model_parameters')
                and component.model_parameters.model_label == folder_name)))
        for folder_name in tuple(next(walk(path))[1]))

    # Add the model folder and data paths
    apply(update_paths, links)

    # Check if the optimized fluence file exists and should not be ignored
    if 'optimized_fluence.npy' in listdir(path) and not ignore_optimum:

        # Load the optimized fluence array
        treatment_plan.fluence_optimizer.optimized_fluence = npload(
            f'{path}/optimized_fluence.npy')

        # Set the copycat flag
        treatment_plan.fluence_optimizer.from_copycat = True

    return treatment_plan
