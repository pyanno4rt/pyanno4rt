"""Instance copycat."""

# Author: Tim Ortkamp

# %% External package import

from glob import glob
from os.path import abspath

from json import load
from numpy import load as npload

# %% Function definition


def copycat(base_class, path, ignore_optimum=False):
    """
    Create a copycat from a treatment plan snapshot.

    Parameters
    ----------
    base_class : :class:`~pyanno4rt.base._treatment_plan.TreatmentPlan`
        The base class from which to create an instance.

    path : str
        Path to the snapshot.

    ignore_optimum : bool
        Indicator for ignoring the optimized fluence file (if available).

    Returns
    -------
    object of class :class:`~pyanno4rt.base._treatment_plan.TreatmentPlan`
        The object used to represent the treatment plan.
    """

    # Get the snapshot path
    copy_path = abspath(path)

    # Open a file stream for the input parameters
    with open(f'{copy_path}/input.json', 'r', encoding='utf-8') as file:

        # Load the parameter dictionaries
        inputs = load(file)

    # Search for the stored data
    snap_patient_data = glob(copy_path+'/patient_data*')
    snap_dose_matrix = glob(copy_path+'/dose_influence_matrix.*')
    snap_optimized_fluence = glob(copy_path+'/optimized_fluence.np*')

    # Check if patient data has been found
    if len(snap_patient_data) == 1:

        # Update the imaging path
        inputs['configuration']['imaging_path'] = snap_patient_data[0]

    # Check if dose-influence matrix data has been found
    if len(snap_dose_matrix) == 1:

        # Update the dose matrix path
        inputs['configuration']['dose_matrix_path'] = snap_dose_matrix[0]

    # Check if relevant optimized fluence data has been found
    if len(snap_optimized_fluence) == 1 and not ignore_optimum:

        # Update the initial fluence vector
        inputs['optimization']['initial_fluence'] = list(npload(
            snap_optimized_fluence[0]))

        # Set the maximum number of iterations to zero
        inputs['optimization']['maximum_iterations'] = 0

    # Loop over the model-based components
    for name, parameters in {
            name: parameters
            for component in inputs['optimization']['components']
            for name, parameters in component.items()
            if 'Outcome' in name}.items():

        # Get the model path
        model_path = f'{copy_path}/{parameters["model"]["label"]}'

        # Update the model path
        parameters['model']['model_path'] = model_path

        # Search for the stored dataset
        snap_model_data = glob(model_path+'/dataset.*')

        # Check if a dataset has been found
        if len(snap_model_data) == 1:

            # Update the dataset path
            parameters['model']['dataset']['data_path'] = snap_model_data[0]

    # Check if the imaging or dose matrix data path is missing
    if None in (
            inputs['configuration']['imaging_path'],
            inputs['configuration']['dose_matrix_path']):

        # Raise an error to indicate missing paths
        raise ValueError(
            "Please specify the configuration data paths (imaging, "
            "dose-influence matrix) in the 'input.json' file!")

    # Initialize the treatment plan instance
    treatment_plan = base_class(**inputs)

    return treatment_plan
