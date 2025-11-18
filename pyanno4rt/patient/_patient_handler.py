"""Patient handling."""

# Author: Tim Ortkamp

# %% External package import

from os.path import splitext

# %% Internal package import

from pyanno4rt.datahub import Datahub
from pyanno4rt.io.patient import DicomHandler, MatHandler

# %% Class definition


class PatientHandler():
    """
    Patient handling class.

    This class provides methods to handle patient imaging data.

    Attributes
    ----------
    computed_tomography : None or dict
        Dictionary with information on the CT images.

    segmentation : None or dict
        Dictionary with information on the segments.
    """

    # Map the path extensions to the handlers
    sources = {
        '': ('DICOM folder', DicomHandler),
        '.mat': ('MATLAB file', MatHandler)}

    def __init__(self):

        # Log a message about the initialization of the class
        Datahub().logger.display_info("Initializing patient handler ...")

        # Initialize the CT and segmentation dictionaries
        self.computed_tomography, self.segmentation = None, None

    def load(
            self,
            path):
        """
        Load the patient imaging data.

        Parameters
        ----------
        path : str
            Path to the patient imaging data.
        """

        # Initialize the datahub
        hub = Datahub()

        # Get the file string and handler
        source, handler = self.sources[splitext(path)[1]]

        # Log a message about the patient imaging data loading
        hub.logger.display_info(
            f"Loading CT and segmentation data from {source} ...")

        # Load the patient imaging data
        self.computed_tomography, self.segmentation = handler().load(path)

        # Store the patient imaging data
        hub.computed_tomography, hub.segmentation = (
            self.computed_tomography, self.segmentation)

    def save(
            self,
            path):
        """
        Save the patient imaging data.

        Parameters
        ----------
        path : str
            Path for storing the patient imaging data.
        """

        # Initialize the datahub
        hub = Datahub()

        # Get the file string and handler
        source, handler = self.sources[splitext(path)[1]]

        # Log a message about the patient imaging data saving
        hub.logger.display_info(
            f"Saving CT and segmentation data to {source} ...")

        # Save the patient imaging data
        handler().save(self.computed_tomography, self.segmentation, path)
