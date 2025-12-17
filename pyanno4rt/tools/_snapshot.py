"""Instance snapshot."""

# Author: Tim Ortkamp

# %% External package import

from os import mkdir
from os.path import abspath, exists, splitext
from shutil import copy

from json import dump
from numpy import save

# %% Internal package import

from pyanno4rt.tools import apply, get_machine_learning_components

# %% Function definition


def snapshot(
        instance, path, include_patient_data=False, include_dose_matrix=False,
        include_model_data=False, include_optimum=False):
    """
    Take a snapshot of a treatment plan.

    Parameters
    ----------
    instance : object of class \
        :class:`~pyanno4rt.base._treatment_plan.TreatmentPlan`
        The object used to represent the treatment plan.

    path : str
        Path for storing the snapshot (folder).

    include_patient_data : bool, default=False
        Indicator for the storage of the patient data, i.e., CT and \
        segmentation data.

    include_dose_matrix : bool, default=False
        Indicator for the storage of the dose-influence matrix.

    include_model_data : bool, default=False
        Indicator for the storage of the outcome model datasets.

    include_optimum : bool, default=False
        Indicator for the storage of the optimized fluence array.
    """

    def export_model_files(model):
        """Export the machine learning model data files."""

        # Get the model folder path
        path = f'{snap_path}/{model.label}'

        # Check if the path does not yet exist
        if not exists(path):

            # Create a new folder
            mkdir(path)

        # Save the model
        model.save(path)

        # Export the configuration file
        # model.export_configuration(include_model_data)

        # Export the hyperparameter file
        # model.export_hyperparameters()

        # Check if the model data should be included and is not None
        if include_model_data:

            # Get the file extension
            _, extension = splitext(model.dataset.path)

            # Copy the model data into a file
            copy(model.dataset.path, f'{path}/dataset{extension}')

    # Get the snapshot folder path
    snap_path = abspath(f'{path}/{instance.configuration.label}')

    # Check if the path does not yet exist
    if not exists(snap_path):

        # Create a new folder
        mkdir(snap_path)

    # Open a file stream for the input parameters
    with open(f'{snap_path}/input.json', 'w', encoding='utf-8') as file:

        # Get the input dictionaries
        input_dictionaries = {
            'configuration': instance.configuration.to_dict(),
            'optimization': instance.optimization.to_dict(),
            'evaluation': instance.evaluation.to_dict()}

        # Dump the input dictionaries
        dump(input_dictionaries, file, sort_keys=False, indent=4)

    # Open a file stream for the log output
    with open(
            f'{snap_path}/{instance.configuration.label}.log', 'w',
            encoding='utf-8') as file:

        # Get the logging stream value
        stream_value = instance.logging.logger.handlers[1].stream.getvalue()

        # Print the stream value to the file
        print(stream_value, file=file)

    # Check if the patient data should be included
    if include_patient_data and instance.state >= 1:

        # Get the file extension
        _, extension = splitext(instance.configuration.imaging_path)

        # Save the patient data
        instance.patient_handler.save(f'{snap_path}/patient_data{extension}')

    # Check if the dose influence matrix data should be included
    if include_dose_matrix and instance.state >= 1:

        # Get the file extension
        _, extension = splitext(instance.configuration.dose_matrix_path)

        # Save the dose-influence matrix
        instance.dose_handler.save_dij(
            f'{snap_path}/dose_influence_matrix{extension}')

    # Check if the optimized fluence array should be included
    if include_optimum and instance.state >= 3:

        # Save the fluence array
        save(f'{snap_path}/optimized_fluence.npy',
             instance.fluence_optimizer.optimized_fluence)

    # Get the machine learning models
    models = tuple(
        component.model for component in get_machine_learning_components(
            instance.plan_handler.components))

    # Export the machine learning model files
    apply(export_model_files, models)
