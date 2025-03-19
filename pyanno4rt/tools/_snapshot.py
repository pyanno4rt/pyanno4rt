"""Instance snapshot."""

# Author: Tim Ortkamp

# %% External package import

from json import dumps
from os.path import abspath, splitext
from zipfile import ZipFile

# %% Internal package import

from pyanno4rt.tools import apply, get_machine_learning_objectives

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
        Directory path for the snapshot (archive).

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

        # Get the model object
        model = data[2]

        # Set the model file paths to the ZIP file
        model.set_file_paths(f'{snap_path}.p4rt')

        # Write the configuration to the ZIP file
        zip_file.writestr(
            f'{data[1]}/configuration.json',
            data=model.export_configuration(include_model_data))

        # Write the hyperparameters to the ZIP file
        zip_file.writestr(
            f'{data[1]}/hyperparameters.json',
            data=model.export_hyperparameters())

        # Write the prediction model to the ZIP file
        zip_file.writestr(
            f'{data[1]}/model.'
            f'{"h5" if "Neural Network" in data[0] else "sav"}',
            data=model.export_model())

        # Write the preprocessor to the ZIP file
        zip_file.writestr(
            f'{data[1]}/preprocessor.sav', data=model.export_preprocessor())

        # Check if the model data should be saved and exists
        if include_model_data and data[3] is not None:

            # Get the file extension
            _, extension = splitext(data[3])

            # Write the model data to the ZIP file
            zip_file.write(data[3], f'{data[1]}/model_data{extension}')

    # Check if any required attribute is missing
    if (any(getattr(instance, attribute) is None for attribute in (
            'logger', 'datahub', 'patient_loader', 'plan_generator',
            'dose_info_generator', 'fluence_optimizer'))
            or instance.datahub.state < 3):

        # Raise an error to indicate a missing attribute
        raise AttributeError(
            "Please configure and optimize the treatment plan before taking a "
            "snapshot!")

    # Build the snapshot folder path
    snap_path = abspath(f"{path}/{instance.configuration.label}")

    # Build a joint dictionary for the plan inputs
    input_dictionaries = {
        'configuration': instance.configuration.to_dict(),
        'optimization': instance.optimization.to_dict(),
        'evaluation': instance.evaluation.to_dict()}

    # Get the machine learning model data
    ml_model_data = tuple(
        (objective.name, objective.model.model_label, objective.model,
         objective.model_parameters.get('data_path'))
        for objective in get_machine_learning_objectives(
                instance.datahub.segmentation))

    # Open a stream to a ZIP file
    with ZipFile(f'{snap_path}.p4rt', mode="w") as zip_file:

        # Write the input parameter dictionary to the ZIP file
        zip_file.writestr(
            'input_parameters.json',
            data=dumps(input_dictionaries, sort_keys=False, indent=4))

        # Write the log file to the ZIP file
        zip_file.writestr(
            f'{instance.datahub.label}.log',
            data=instance.logger.logger.handlers[1].stream.getvalue())

        # Export the data for the machine learning model(s)
        apply(export_model_files, ml_model_data)

        # Check if the patient data should be saved
        if include_patient_data:

            # Get the file extension
            _, extension = splitext(instance.configuration.imaging_path)

            # Write the patient data to the ZIP file
            zip_file.write(
                instance.configuration.imaging_path,
                f'patient_data{extension}')

        # Check if the dose-influence matrix should be saved
        if include_dose_matrix:

            # Get the file extension
            _, extension = splitext(instance.configuration.dose_matrix_path)

            # Write the dose-influence matrix to the ZIP file
            zip_file.write(
                instance.configuration.dose_matrix_path,
                f'dose_influence_matrix{extension}')

        # Check if the optimized fluence array should be saved
        if include_optimum:

            # Write the optimized fluence array to the ZIP file
            zip_file.writestr(
                'optimized_fluence.npy',
                data=instance.datahub.optimization['optimized_fluence'])

        # Test the integrity of the ZIP file
        zip_file.testzip()
