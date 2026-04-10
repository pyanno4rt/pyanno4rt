"""Instance snapshot."""

# Author: Tim Ortkamp

# %% External package import

from os import mkdir
from os.path import exists, splitext

from json import dump

# %% Function definition


def snapshot(
        instance, path, include_patient_data=False, include_dose_matrix=False,
        include_model_data=False, include_optimum=False, include_tracks=False,
        anonymize=False):
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

    include_tracks : bool, default=False
        Indicator for the storage of the component tracker.

    anonymize : bool, default=False
        Indicator for the anonymization of the file paths.
    """

    # Check if the path does not yet exist
    if not exists(path):

        # Create a new folder
        mkdir(path)

    # Open a file stream for the logging output
    with open(
            f'{path}/{instance.configuration.label}.log', 'w',
            encoding='utf-8') as file:

        # Get the logging stream value
        stream_value = instance.logging.logger.handlers[1].stream.getvalue()

        # Print the stream value to the file
        print(stream_value, file=file)

    # Check if the patient data should be included
    if include_patient_data and instance.state >= 1:

        # Get the file extension
        _, extension = splitext(instance.configuration.imaging_path)

        # Set the file path
        pat_path = f'{path}/patient_data{extension}'

        # Save the patient data
        instance.patient_handler.save(pat_path)

        # Update the imaging path
        instance.configuration.imaging_path = pat_path

    # Check if the dose influence matrix data should be included
    if include_dose_matrix and instance.state >= 1:

        # Get the file extension
        _, extension = splitext(instance.configuration.dose_matrix_path)

        # Set the file path
        dij_path = f'{path}/dose_influence_matrix{extension}'

        # Save the dose-influence matrix
        instance.dose_handler.save_dij(dij_path)

        # Update the dose-influence matrix path
        instance.configuration.dose_matrix_path = dij_path

    # Check if imaging and dose matrix path should be anonymized
    if anonymize:

        # Set the paths to None
        instance.configuration.imaging_path = None
        instance.configuration.dose_matrix_path = None

    # Check if the instance has already been modeled
    if instance.state >= 2 and instance.data_model_handler is not None:

        # Loop over the machine learning models
        for model in instance.data_model_handler.models:

            # Set the model path
            model_path = f'{path}/{model.label}'

            # Check if the path does not yet exist
            if not exists(model_path):

                # Create a new folder
                mkdir(model_path)

            # Save the model
            model.save(model_path)

            # Update the model path
            model.arguments['model_path'] = model_path

            # Check if the model data path should be anonymized
            if anonymize:

                # Set the data path to None
                model.dataset.arguments['data_path'] = None

            # Check if the model data should be included
            if include_model_data:

                # Get the file extension
                _, extension = splitext(model.dataset.path)

                # Set the file path
                data_path = f'{path}/dataset{extension}'

                # Save the dataset
                model.dataset.save(data_path)

                # Update the model data path
                model.dataset.arguments['data_path'] = data_path

    # Check if the optimized fluence array should be included
    if include_optimum and instance.state >= 3:

        # Save the fluence array
        instance.fluence_optimizer.save_fluence(
            f'{path}/optimized_fluence.npy')

    # Check if the component tracker should be included
    if include_tracks and instance.state >= 3:

        # Save the component tracker
        instance.fluence_optimizer.problem.save_tracker(
            f'{path}/tracker.json')

    # Open a file stream for the input parameters
    with open(f'{path}/input.json', 'w', encoding='utf-8') as file:

        # Get the input dictionaries
        input_dictionaries = {
            'configuration': instance.configuration.to_dict(),
            'optimization': instance.optimization.to_dict(),
            'evaluation': instance.evaluation.to_dict()}

        # Dump the input dictionaries
        dump(input_dictionaries, file, sort_keys=False, indent=4)
