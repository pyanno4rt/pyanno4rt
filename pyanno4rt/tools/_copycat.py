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
        Path to the snapshot (folder).

    ignore_optimum : bool
        Indicator for ignoring the optimal fluence file (if available).

    Returns
    -------
    object of class :class:`~pyanno4rt.base._treatment_plan.TreatmentPlan`
        The object used to represent the treatment plan.
    """

    # Get the copycat folder path
    copy_path = abspath(path)

    # Open a file stream for the input parameters
    with open(f'{copy_path}/input.json', 'r', encoding='utf-8') as file:

        # Load the parameter dictionaries
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

    # Get the folder/model links
    model_components = (
        (f'{path}/{component.model.label}', component)
        for component in inputs['optimization']['components']
        if 'Outcome' in next(iter(component)))

    #
    for path, component in model_components:

        # Update the model path
        component.model.path = path

        #
        component.model.load()

        #
        snap_model_data = glob(path+'/dataset*')

        #
        if len(snap_model_data) == 1:

            #
            component.model.dataset.path = snap_model_data[0]

    # Initialize the treatment plan instance
    treatment_plan = base_class(**inputs)

    return treatment_plan
