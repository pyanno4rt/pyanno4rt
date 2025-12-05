"""Instance copycat."""

# Author: Tim Ortkamp

# %% External package import

from glob import glob
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
        Path to the snapshot (folder).

    ignore_optimum : bool
        Indicator for ignoring the optimal fluence file (if available).

    Returns
    -------
    object of class :class:`~pyanno4rt.base._treatment_plan.TreatmentPlan`
        The object used to represent the treatment plan.
    """

    # def update_model_paths(inputs):
    #     """Update the model folder and data path for a component."""

    #     # Get the model path and the component
    #     path, component = inputs

    #     # Update the model path
    #     component.model_parameters.model_folder_path = path

    #     # Reset the data path
    #     component.model_parameters.data_path = None

    #     # Loop over the model path files
    #     for filename in listdir(path):

    #         # Check if the data file exists
    #         if 'model_data' in filename:

    #             # Update the data path
    #             component.model_parameters.data_path = f'{path}/{filename}'

    # Open a file stream
    with open(f'{path}/input.json', 'r', encoding='utf-8') as file:

        # Load the input parameter dictionaries
        inputs = load(file)

    # Search for the stored data
    snap_patient_data = glob(path+'/patient_data*')
    snap_dose_matrix = glob(path+'/dose_influence_matrix.*')
    snap_optimized_fluence = glob(path+'/optimized_fluence.npy')

    # Check if patient data has been found
    if len(snap_patient_data) == 1:

        # Update the imaging path
        inputs['configuration']['imaging_path'] = snap_patient_data[0]

    # Check if dose-influence matrix data has been found
    if len(snap_dose_matrix) == 1:

        # Update the dose matrix path
        inputs['configuration']['dose_matrix_path'] = snap_dose_matrix[0]

    # Check if an optimized fluence has been found and should not be ignored
    if len(snap_optimized_fluence) == 1 and not ignore_optimum:

        # Update the initial fluence vector
        inputs['optimization']['initial_fluence'] = list(npload(
            snap_optimized_fluence[0]))

        # Set the maximum number of iterations to zero
        inputs['optimization']['maximum_iterations'] = 0

    # Initialize the treatment plan instance
    treatment_plan = base_class(**inputs)

    # # Get the folder/model links
    # links = (
    #     (f'{path}/{folder_name}', next(
    #         component for component in treatment_plan.optimization.components
    #         if (hasattr(component, 'model_parameters')
    #             and component.model_parameters.model_label == folder_name)))
    #     for folder_name in tuple(next(walk(path))[1]))

    # # Add the model folder and data paths
    # apply(update_model_paths, links)

    return treatment_plan
