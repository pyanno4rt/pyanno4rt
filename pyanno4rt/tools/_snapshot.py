"""Instance snapshot."""

# Author: Tim Ortkamp

# %% External package import

from os import mkdir
from os.path import abspath, exists, splitext

from json import dump

# %% Function definition


def snapshot(
        instance, path, include_patient_data=False, include_dose_matrix=False,
        include_model_data=False, include_fluence=False):
    """
    Take a snapshot of a treatment plan.

    Parameters
    ----------
    instance : object of class \
        :class:`~pyanno4rt.base._treatment_plan.TreatmentPlan`
        The object used to represent the treatment plan.

    path : str
        Path for storing the snapshot.

    include_patient_data : bool, default=False
        Indicator for the storage of the patient data, i.e., CT and \
        segmentation data.

    include_dose_matrix : bool, default=False
        Indicator for the storage of the dose-influence matrix.

    include_model_data : bool, default=False
        Indicator for the storage of the outcome model datasets.

    include_fluence : bool, default=False
        Indicator for the storage of the optimized fluence array.
    """

    # Get the snapshot path
    snap_path = abspath(f'{path}/{instance.configuration.label}')

    # Check if the path does not yet exist
    if not exists(snap_path):

        # Create a new folder
        mkdir(snap_path)

    # Open a file stream for the logging output
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

        # Get the file path
        path = f'{snap_path}/patient_data{extension}'

        # Save the patient data
        instance.patient_handler.save(path)

        # Update the imaging path
        instance.configuration.imaging_path = path

    # Check if the dose influence matrix data should be included
    if include_dose_matrix and instance.state >= 1:

        # Get the file extension
        _, extension = splitext(instance.configuration.dose_matrix_path)

        # Get the file path
        path = f'{snap_path}/dose_influence_matrix{extension}'

        # Save the dose-influence matrix
        instance.dose_handler.save_dij(path)

        # Update the dose-influence matrix path
        instance.configuration.dose_matrix_path = path

    # Check if the instance has already been modeled
    if instance.state >= 2:

        # Loop over the machine learning models
        for model in instance.data_model_handler.models:

            # Get the model path
            path = f'{snap_path}/{model.label}'

            # Check if the path does not yet exist
            if not exists(path):

                # Create a new folder
                mkdir(path)

            # Save the model
            model.save(path)

            # Check if the model data should be included
            if include_model_data:

                # Get the file extension
                _, extension = splitext(model.dataset.path)

                # Get the file path
                data_path = f'{path}/dataset{extension}'

                # Save the dataset
                model.dataset.save(data_path)

                # Update the model data path
                model.dataset.inputs['path'] = data_path

    # Check if the optimized fluence array should be included
    if include_fluence and instance.state >= 3:

        # Save the fluence array
        instance.fluence_optimizer.save_fluence(
            f'{snap_path}/optimized_fluence.npy')

    # Open a file stream for the input parameters
    with open(f'{snap_path}/input.json', 'w', encoding='utf-8') as file:

        # Get the input dictionaries
        input_dictionaries = {
            'configuration': instance.configuration.to_dict(),
            'optimization': instance.optimization.to_dict(),
            'evaluation': instance.evaluation.to_dict()}

        # Dump the input dictionaries
        dump(input_dictionaries, file, sort_keys=False, indent=4)
