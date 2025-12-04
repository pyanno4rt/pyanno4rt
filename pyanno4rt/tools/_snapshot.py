"""Instance snapshot."""

# Author: Tim Ortkamp

# %% External package import

from os import mkdir
from os.path import abspath, exists, splitext
from shutil import copy

from json import dump
from numpy import save

# %% Internal package import

from pyanno4rt.tools import apply

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
        The object from which to take a snapshot.

    path : str
        Directory path for the snapshot (folder).

    include_patient_data : bool, default=False
        Indicator for the storage of the external patient data, i.e., \
        CT and segmentation data.

    include_dose_matrix : bool, default=False
        Indicator for the storage of the dose-influence matrix.

    include_model_data : bool, default=False
        Indicator for the storage of the outcome model-related datasets.

    include_optimum : bool, default=False
        Indicator for the storage of the optimized fluence array.

    Raises
    ------
    AttributeError
        If the treatment plan has not been configured and optimized yet.
    """

    def export_model_files(data):
        """Export the machine learning model data files."""

        # Build the model folder path
        model_path = f'{snap_path}/{data[0]}'

        # Check if the path does not yet exist
        if not exists(model_path):

            # Create a new folder
            mkdir(model_path)

        # Get the model object
        model = data[1]

        # Set the file paths in the model
        model.set_file_paths(model_path)

        # Export the configuration file
        model.export_configuration(include_model_data)

        # Export the hyperparameter file
        model.export_hyperparameters()

        # Export the prediction model file
        model.export_model()

        # Export the preprocessor file
        model.export_preprocessor()

        # Check if the model data should be included and is not None
        if include_model_data and data[2] is not None:

            # Get the file extension
            _, extension = splitext(data[2])

            # Copy the model data into a file
            copy(data[2], f'{model_path}/model_data{extension}')

    # Check if any required attribute is missing
    if (any(getattr(instance, attribute) is None for attribute in (
            'logging', 'datahub', 'patient_handler', 'plan_handler',
            'dose_handler', 'fluence_optimizer'))
            or instance.state < 3):

        # Raise an error to indicate a missing attribute
        raise AttributeError(
            "Please configure and optimize the treatment plan before taking a "
            "snapshot!")

    # Build the snapshot folder path
    snap_path = abspath(f'{path}/{instance.configuration.label}')

    # Check if the path does not yet exist
    if not exists(snap_path):

        # Create a new folder
        mkdir(snap_path)

    # Open a file stream for the input parameters
    with open(f'{snap_path}/input_parameters.json', 'w',
              encoding='utf-8') as file:

        # Get the input dictionaries
        input_dictionaries = {
            'configuration': instance.configuration.to_dict(),
            'optimization': instance.optimization.to_dict(),
            'evaluation': instance.evaluation.to_dict()}

        # Dump the input dictionaries
        dump(input_dictionaries, file, sort_keys=False, indent=4)

    # Open a file stream for the log output
    with open(f'{snap_path}/{instance.configuration.label}.log', 'w',
              encoding='utf-8') as file:

        # Get the logging stream value
        stream_value = instance.logging.logger.handlers[1].stream.getvalue()

        # Print the stream value to the file
        print(stream_value, file=file)

    # Get the machine learning model data
    ml_model_data = tuple((
        component.model.model_label, component.model,
        component.model_parameters.data_path)
        for component in (
            instance.plan_handler.get_components('objective', 'ml', False)
            + instance.plan_handler.get_components('constraint', 'ml', False)))

    # Export the machine learning model files
    apply(export_model_files, ml_model_data)

    # Check if the patient data should be included
    if include_patient_data:

        # Get the file extension
        _, extension = splitext(instance.configuration.imaging_path)

        # Copy the patient data into a file
        copy(instance.configuration.imaging_path,
             f'{snap_path}/patient_data{extension}')

    # Check if the dose influence matrix data should be included
    if include_dose_matrix:

        # Get the file extension
        _, extension = splitext(instance.configuration.dose_matrix_path)

        # Copy the matrix into a file
        copy(instance.configuration.dose_matrix_path,
             f'{snap_path}/dose_influence_matrix{extension}')

    # Check if the optimized fluence array should be included
    if include_optimum:

        # Save the fluence array
        save(f'{snap_path}/optimized_fluence.npy',
             instance.fluence_optimizer.optimized_fluence)
