"""Empty dataset generation."""

# Author: Tim Ortkamp

# %% External package import

from functools import partial
from json import load as jload

# %% Internal package import

from pyanno4rt.datahub import Datahub
from pyanno4rt.learning_model.features import feature_map
from pyanno4rt.tools import identity

# %% Class definition


class EmptyDataGenerator():
    """
    Empty dataset generation class.

    This class provides methods to set up a default empty base dataset.

    Parameters
    ----------
    model_label : str
        Label for the machine learning model.

    model_folder_path : None or str
        Path to a folder for loading an external model.

    Attributes
    ----------
    model_label : str
        See 'Parameters'.

    model_folder_path : None or str
        See 'Parameters'.
    """

    def __init__(
            self,
            model_label,
            model_folder_path):

        # Log a message about the initialization of the class
        Datahub().logger.display_info(
            f"Initializing data generator for '{model_label}' model from "
            "empty base dataset ...")

        # Get the instance attributes from the arguments
        self.model_label = model_label
        self.model_folder_path = model_folder_path

    def generate(self):
        """
        Generate the data information.

        Parameters
        ----------
        *args : tuple
            Tuple with additional (non-keyworded) arguments.

        Returns
        -------
        dict
            Dictionary with the default data information.
        """

        # Initialize the datahub
        hub = Datahub()

        # Get the configuration file path
        configuration_path = ''.join(
            (self.model_folder_path, '/configuration.json'))

        # Open a file stream
        with open(configuration_path, 'r', encoding='utf-8') as file:

            # Load the configuration
            configuration = jload(file)

        # Generate the data information dictionary
        data_information = {
            'feature_names': configuration['feature_names'],
            'feature_scales': configuration['feature_scales'],
            'label_name': configuration['label_name'],
            'label_viewpoint': configuration['label_viewpoint'],
            'label_bounds': configuration['label_bounds'],
            'time_variable_name': configuration['time_variable_name'],
            'feature_statics': configuration['feature_statics'],
            'feature_definitions': configuration['feature_definitions'],
            'tune_folds': configuration['tune_folds'],
            'oof_folds': configuration['oof_folds'],
            'number_of_samples': configuration['number_of_samples']
            }

        # Enter the data information dictionary into the datahub
        hub.datasets |= {self.model_label: data_information}

        # Generate the feature map dictionary
        feature_map_dict = self.create_map(
            data_information['feature_definitions'])

        # Enter the feature map into the datahub
        hub.feature_maps |= {self.model_label: feature_map_dict}

        return data_information, feature_map_dict

    def create_map(
            self,
            definitions):
        """
        Create the feature map.

        Parameters
        ----------
        definitions : dict
            Dictionary with the mappings of feature names, segments and \
            string functions.

        Returns
        -------
        dict
            Dictionary with the mappings of feature names, segments and \
            computation/differentiation functions.
        """

        def get_single_definition(key):
            """Get the mapping for a single definition."""

            # Get the feature definition as string
            definition = feature_map.get(definitions[key]['function'])

            # Get the argument of the feature definition
            args = definitions[key]['argument']

            # Check if a definition and no value have been passed
            if definition and not definitions[key]['value']:

                # Return the dosiomic/radiomic feature map
                return {key: {
                    'segment': definitions[key]['segment'],
                    'class': definition.feature_class,
                    'computation': (
                        methods[args is None](definition.compute, args)),
                    'differentiation': (
                        methods[args is None](definition.differentiate, args)
                        if definition.feature_class == 'Dosiomics' else None)}}

            else:

                # Return the static feature map
                return {key: {
                    'segment': None,
                    'class': 'Statics',
                    'computation': None,
                    'differentiation': None}}

        # Create a boolean mapping to the internal functions
        methods = {True: identity, False: partial}

        return {key: value
                for item in map(get_single_definition, definitions.keys())
                for key, value in item.items()}
