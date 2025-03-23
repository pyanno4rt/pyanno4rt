"""Image dataset generation."""

# Author: Tim Ortkamp

# %% External package import

from json import load as jload

# %% Internal package import

from pyanno4rt.datahub import Datahub

# %% Class definition


class ImageDataGenerator():
    """
    Image dataset generation class.

    This class provides methods to set up an image base dataset.

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

        # Get the configuration file path
        configuration_path = f'{self.model_folder_path}/configuration.json'

        # Generate the data information dictionary
        data_information = {
            'feature_names': jload(open(
                configuration_path, 'r', encoding='utf-8'))['feature_names'],
            'label_name': jload(open(
                configuration_path, 'r', encoding='utf-8'))['label_name']}

        # Enter the data information dictionary into the datahub
        Datahub().datasets |= {self.model_label: data_information}

        return data_information
