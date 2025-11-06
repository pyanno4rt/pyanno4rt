"""Patient loading."""

# Author: Tim Ortkamp

# %% External package import

from os.path import splitext

# %% Internal package import

from pyanno4rt.datahub import Datahub
from pyanno4rt.io import DicomHandler, MatHandler

# %% Class definition


class PatientLoader():
    """
    Patient loading class.

    This class provides methods to load patient data from different input \
    formats and generate the CT and segmentation dictionaries.

    Parameters
    ----------
    imaging_path : str
        Path to the CT and segmentation data.

    Attributes
    ----------
    imaging_path : str
        See 'Parameters'.
    """

    def __init__(
            self,
            imaging_path):

        # Log a message about the initialization of the class
        Datahub().logger.display_info("Initializing patient loader ...")

        # Get the imaging path from the arguments
        self.imaging_path = imaging_path

    def load(self):
        """Load the patient data."""

        # Initialize the datahub
        hub = Datahub()

        # Map the path extensions to the sources and handlers
        sources = {
            '': ('DICOM folder', DicomHandler),
            '.mat': ('MATLAB file', MatHandler)}

        # Get the string and handler from the extension
        source, handler = sources[splitext(self.imaging_path)[1]]

        # Log a message about the import of the patient imaging data
        hub.logger.display_info(
            f"Importing CT and segmentation data from {source} ...")

        # Enter the patient data into the datahub
        hub.computed_tomography, hub.segmentation = handler().extract(
            self.imaging_path)
