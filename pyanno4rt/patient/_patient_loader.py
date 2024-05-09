"""Patient loading."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from os.path import splitext

# %% Internal package import

from pyanno4rt.datahub import Datahub
from pyanno4rt.patient.import_functions import (
    import_from_dcm, import_from_mat, import_from_p)

# %% Class definition


class PatientLoader():
    """
    Patient loading class.

    This class provides methods to load patient data from different input \
    formats and generate the computed tomography (CT) and segmentation \
    dictionaries.

    Parameters
    ----------
    imaging_path : str
        Path to the CT and segmentation data.

    target_imaging_resolution : None or list
        Imaging resolution for post-processing interpolation of the CT and \
        segmentation data.

    Attributes
    ----------
    imaging_path : str
        See 'Parameters'.

    target_imaging_resolution : None or list
        See 'Parameters'.
    """

    def __init__(
            self,
            imaging_path,
            target_imaging_resolution):

        # Log a message about the initialization of the class
        Datahub().logger.display_info("Initializing patient loader ...")

        # Get the instance attributes from the arguments
        self.imaging_path = imaging_path
        self.target_imaging_resolution = target_imaging_resolution

    def load(self):
        """Load the patient data from the path."""

        # Initialize the datahub
        hub = Datahub()

        # Map the path extensions to the sources and import functions
        sources = {'': ('DICOM folder', import_from_dcm),
                   '.mat': ('MATLAB file', import_from_mat),
                   '.p': ('Python file', import_from_p)}

        # Get the source and import function from the extension
        source, importer = sources[splitext(self.imaging_path)[1]]

        # Log a message about the import of the patient imaging data
        hub.logger.display_info(
            f"Importing CT and segmentation data from {source} ...")

        # Enter the patient imaging data into the datahub
        hub.computed_tomography, hub.segmentation = importer(
            self.imaging_path, self.target_imaging_resolution)
