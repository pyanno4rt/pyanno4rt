"""Instance copycat."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from itertools import product
from json import load
from os import listdir, walk
from os.path import basename

# %% Internal package import

from pyanno4rt.tools import apply

# %% Function definition


def copycat(base_class, path):
    """
    Create a copycat from a treatment plan snapshot.

    Parameters
    ----------
    base_class : class ``TreatmentPlan``
        Treatment plan class from which to create the instance.

    path : string
        Directory path for the snapshot folder.

    Returns
    -------
    object of class ``TreatmentPlan``
        Treatment plan instance.
    """

    def add_model_paths(inputs):
        """Add the model folder and data path to each model component."""

        def edit(component):
            """Edit a single component."""

            # Check if the component has model parameters
            # and if the model label is equal to the folder name
            if ('model_parameters' in component['parameters']
                and component['parameters']['model_parameters'][
                    'model_label'] == basename(inputs[0])):

                # Overwrite the model folder path of the component
                component['parameters']['model_parameters'][
                    'model_folder_path'] = inputs[0]

                # Loop over the model path files
                for filename in listdir(inputs[0]):

                    # Check if the model data file exists
                    if 'model_data' in filename:

                        # Overwrite the model data path
                        component['parameters']['model_parameters'][
                            'data_path'] = ''.join((inputs[0], '/', filename))

        # Get the component(s)
        component = treatment_plan.optimization['components'][inputs[1]][1]

        # Check if the component is a list
        if isinstance(component, list):

            # Apply the editing function to each element in the component
            apply(edit, component)

        # Else, check if the component is a dict
        elif isinstance(component, dict):

            # Edit the component
            edit(component)

    # Open a file stream
    with open(''.join((path, '/input_parameters.json')), 'r',
              encoding='utf-8') as file:

        # Load the input parameter dictionaries
        input_parameters = load(file)

    # Loop over the path files
    for filename in listdir(path):

        # Check if the patient data file exists
        if 'patient_data' in filename:

            # Overwrite the imaging path
            input_parameters['configuration']['imaging_path'] = (
                ''.join((path, '/', filename)))

        # Check if the dose influence matrix file exists
        elif 'dose_influence_matrix' in filename:

            # Overwrite the dose path
            input_parameters['configuration']['dose_path'] = (
                ''.join((path, '/', filename)))

    # Initialize the treatment plan instance from the input parameters
    treatment_plan = base_class(**input_parameters)

    # Get the model folder paths
    model_paths = (''.join((path, '/', folder_name))
                   for folder_name in tuple(next(walk(path))[1]))

    # Add the model folder and data paths
    apply(add_model_paths,
          product(model_paths, (*treatment_plan.optimization['components'],)))

    return treatment_plan
