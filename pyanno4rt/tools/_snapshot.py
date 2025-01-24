"""Instance snapshot."""

# Author: Tim Ortkamp

# %% External package import

from json import dump
from numpy import save
from os import mkdir
from os.path import abspath, exists, splitext
from shutil import copy

# %% Internal package import

from pyanno4rt.tools import apply, get_machine_learning_objectives

# %% Function definition


def snapshot(instance, path, include_patient_data=False,
             include_dose_matrix=False, include_model_data=False,
             include_optimum=False):
    """
    Take a snapshot of a treatment plan.

    Parameters
    ----------
    instance : object of class from :mod:`~pyanno4rt.base`
        The base treatment plan class from which to take a snapshot.

    path : str
        Directory path for the snapshot (folder).

        .. note:: If the specified path does not reference an existing \
            folder, one is created automatically.

    include_patient_data : bool, default=False
        Indicator for the storage of the external patient data, i.e., \
        computed tomography and segmentation data.

    include_dose_matrix : bool, default=False
        Indicator for the storage of the dose-influence matrix.

    include_model_data : bool, default=False
        Indicator for the storage of the outcome model-related dataset(s).

    include_optimum : bool, default=False
        Indicator for the storage of the optimized fluence.

    Raises
    ------
    AttributeError
        If the treatment plan instance has not been configured yet.
    """

    # Check if any required attribute is missing
    if (any(getattr(instance, attribute) is None for attribute in
            ('logger', 'datahub', 'input_checker', 'patient_loader',
             'plan_generator', 'dose_info_generator', 'fluence_optimizer'))
            or instance.datahub.state < 3):

        # Raise an error to indicate a missing attribute
        raise AttributeError("Please configure and optimize the treatment "
                             "plan before taking a snapshot!")

    def dict_path_to_absolute(search_key, dictionary):
        """Search a path key and convert the value into an absolute path."""

        # Loop over the dictionary items
        for key, value in dictionary.items():

            # Check if the current key has been searched
            if key == search_key:

                # Check if the value is None
                if value is None:

                    # Set the path to None
                    dictionary[key] = value

                else:

                    # Convert the path into an absolute value
                    dictionary[key] = abspath(value)

            # Check if the value is a dictionary
            elif isinstance(value, dict):

                # Loop recursively over the function output
                dict_path_to_absolute(search_key, value)

            # Else, check if the value is a list
            elif isinstance(value, list):

                # Loop over the list elements
                for element in value:

                    # Check if the list element is a dictionary
                    if isinstance(element, dict):

                        # Loop recursively over the function output
                        dict_path_to_absolute(search_key, element)

        return dictionary

    def save_ml_model(data):
        """Create and save the machine learning model data files."""

        # Build the model folder path
        model_path = f'{snap_path}/{data[0]}'

        # Check if the model folder does not yet exist
        if not exists(model_path):

            # Create a new folder for the model files
            mkdir(model_path)

        # Get the model object
        model = data[1]

        # Set the file path to the current location
        model.set_file_paths(model_path)

        # Write the preprocessor to a file
        model.write_preprocessor_to_file(model.preprocessor)

        # Write the prediction model to a file
        model.write_model_to_file(model.prediction_model)

        # Write the configuration to a file
        model.write_configuration_to_file(
            model.configuration, include_model_data)

        # Write the hyperparameters to a file
        model.write_hyperparameters_to_file(model.hyperparameters)

        # Check if the model data should be saved and exists
        if include_model_data and data[2]:

            # Get the file extension
            _, extension = splitext(data[2])

            # Copy the raw data set into a new file
            copy(data[2], f'{model_path}/model_data{extension}')

    # Build the snapshot folder path
    snap_path = abspath(f"{path}/{instance.configuration['label']}")

    # Check if the folder path does not already exists
    if not exists(snap_path):

        # Create a new folder for the instance files
        mkdir(snap_path)

    # Build a joint dictionary for the plan inputs
    input_dictionaries = {'configuration': instance.configuration,
                          'optimization': instance.optimization,
                          'evaluation': instance.evaluation}

    # Get the machine learning model data
    ml_model_data = tuple((objective.model.model_label, objective.model,
                           objective.model_parameters.get('data_path'))
                          for objective in get_machine_learning_objectives(
                              instance.datahub.segmentation))

    # Check if machine learning model data exists
    if len(ml_model_data) > 0:

        # Convert the data file paths into absolute paths
        input_dictionaries['optimization'] = dict_path_to_absolute(
            'data_path', input_dictionaries['optimization'])

    # Convert the configuration file paths into absolute paths
    for key in ('imaging_path', 'dose_matrix_path'):
        input_dictionaries['configuration'] = dict_path_to_absolute(
            key, input_dictionaries['configuration'])

    # Open a file stream
    with open(f'{snap_path}/input_parameters.json', 'w',
              encoding='utf-8') as file:

        # Dump the input dictionaries to the file
        dump(input_dictionaries, file, sort_keys=False, indent=4)

    # Get the object stream value from the logger
    stream_value = instance.logger.logger.handlers[1].stream.getvalue()

    # Open a file stream
    with open(f'{snap_path}/{instance.datahub.label}.log', 'w',
              encoding='utf-8') as file:

        # Print the stream value to the file
        print(stream_value, file=file)

    # Save the data for the machine learning model(s)
    apply(save_ml_model, ml_model_data)

    # Check if the patient data should be saved
    if include_patient_data:

        # Get the file extension
        _, extension = splitext(instance.configuration['imaging_path'])

        # Copy the input file into a new file
        copy(instance.configuration['imaging_path'],
             f'{snap_path}/patient_data{extension}')

    # Check if the dose influence matrix data should be saved
    if include_dose_matrix:

        # Get the file extension
        _, extension = splitext(instance.configuration['dose_path'])

        # Copy the input file into a new file
        copy(instance.configuration['dose_path'],
             f'{snap_path}/dose_influence_matrix{extension}')

    # Check if the optimized fluence should be saved
    if include_optimum:

        # Save the optimized fluence to a new file
        save(f'{snap_path}/optimized_fluence.npy',
             instance.datahub.optimization['optimized_fluence'])
