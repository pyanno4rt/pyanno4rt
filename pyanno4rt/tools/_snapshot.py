"""Instance snapshots."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from json import dump
from os import mkdir
from os.path import exists, splitext
from shutil import copy

# %% Internal package import

from pyanno4rt.tools import apply, get_model_objectives

# %% Function definition


def snapshot(instance, path, include_patient_data=False,
             include_dose_matrix=False, include_model_data=False):
    """
    Take a snapshot of a treatment plan instance.

    Parameters
    ----------
    instance : object of class `TreatmentPlan`
        Instance of the class `TreatmentPlan`.

    path : string
        Path for the storage of the snapshot folder.

    include_patient_data : bool, default = False
        Indicator for the storage of the patient data (computed tomography,
        segmentation).

    include_dose_matrix : bool, default = False
        Indicator for the storage of the dose-influence matrix.

    include_model_data : bool, default = False
        Indicator for the storage of each model-specific dataset.

    Notes
    -----
    If the function parameter ``path`` does not reference an existing folder, \
    it will automatically create one.
    """

    # Check if any required attribute is missing
    if not all(hasattr(instance, attribute) for attribute in
               ('logger', 'datahub')):

        # Raise an error to indicate a missing attribute
        raise AttributeError(
            "Please configure the treatment plan before taking a snapshot!")

    def save_model_data(data):
        """Create the data files for a model."""

        # Build the path for the respective model folder
        model_file_path = ''.join((snap_path, '/', data[0]))

        # Check if the model folder does not yet exist
        if not exists(model_file_path):

            # Create a new folder for the model files
            mkdir(model_file_path)

        # Get the model object
        model = data[1]

        # Set the file path to the current location
        model.set_file_paths(model_file_path)

        # Write the prediction model to a file
        model.write_model_to_file(model.prediction_model)

        # Write the configuration to a file
        model.write_configuration_to_file(model.configuration)

        # Write the hyperparameters to a file
        model.write_hyperparameters_to_file(model.hyperparameters)

        # Check if the model data should be saved
        if include_model_data:

            # Get the file extension
            _, extension = splitext(data[2])

            # Copy the raw data set into a new file
            copy(data[2],
                 ''.join((model_file_path, '/', 'model_data', extension)))

    # Build the snapshot folder path
    snap_path = ''.join((path, '/', instance.configuration['label']))

    # Check if the folder path does not already exists
    if not exists(snap_path):

        # Create a new folder for the instance files
        mkdir(snap_path)

    # Build a joint dictionary for the plan inputs
    input_dictionaries = {'configuration': instance.configuration,
                          'optimization': instance.optimization,
                          'evaluation': instance.evaluation}

    # Open a file stream
    with open(''.join((snap_path, '/', 'input_parameters.json')), 'w',
              encoding="utf-8") as file:

        # Dump the input dictionaries to the file
        dump(input_dictionaries, file, sort_keys=False, indent=4)

    # Get the object stream value
    stream_value = instance.logger.logger.handlers[1].stream.getvalue()

    # Open a file stream
    with open(''.join((snap_path, '/', instance.datahub.label, '.log')), 'w',
              encoding="utf-8") as file:

        # Print the stream value to the file
        print(stream_value, file=file)

    # Get the learning model data
    model_data = (
        (objective.model.model_label, objective.model,
         objective.model_parameters['data_path'])
        for objective in get_model_objectives(instance.datahub.segmentation))

    # Save the data for each model
    apply(save_model_data, model_data)

    # Check if the patient data should be saved
    if include_patient_data:

        # Get the file extension
        _, extension = splitext(instance.configuration['imaging_path'])

        # Copy the input file into a new file
        copy(instance.configuration['imaging_path'],
             ''.join((snap_path, '/', 'patient_data', extension)))

    # Check if the dose influence matrix data should be saved
    if include_dose_matrix:

        # Get the file extension
        _, extension = splitext(instance.configuration['dose_path'])

        # Copy the input file into a new file
        copy(instance.configuration['dose_path'],
             ''.join((snap_path, '/', 'dose_influence_matrix', extension)))
