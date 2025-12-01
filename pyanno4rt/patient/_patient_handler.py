"""Patient handling."""

# Author: Tim Ortkamp

# %% External package import

from os.path import splitext

from functools import reduce
from numpy import (
    ravel_multi_index, setdiff1d, union1d, unravel_index, where, zeros)
from scipy.ndimage import zoom

# %% Internal package import

from pyanno4rt.io.patient import DicomHandler, MatHandler
from pyanno4rt.logging import get_logger
from pyanno4rt.tools import apply

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
        get_logger().info("Initializing patient handler ...")

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

        # Get the file string and handler
        source, handler = self.sources[splitext(path)[1]]

        # Log a message about the patient imaging data loading
        get_logger().info(
            "Loading CT and segmentation data from %s ...", source)

        # Load the patient imaging data
        self.computed_tomography, self.segmentation = handler().load(path)

        # Remove the segment overlaps
        self.remove_overlap()

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

        # Get the file string and handler
        source, handler = self.sources[splitext(path)[1]]

        # Log a message about the patient imaging data saving
        get_logger().info("Saving CT and segmentation data to %s ...", source)

        # Save the patient imaging data
        handler().save(self.computed_tomography, self.segmentation, path)

    def remove_overlap(self):
        """Remove overlaps between segments."""

        def remove_segment_overlap(reference):
            """Remove the overlap from a reference segment."""

            # Get the indices from all higher prioritized segments
            superior_indices = (
                segmentation[segment]['raw_indices']
                for segment in segmentation
                if (segmentation[segment]['parameters']['priority']
                    < segmentation[reference]['parameters']['priority']))

            # Get the overlap-free (prioritized) indices
            segmentation[reference]['prioritized_indices'] = setdiff1d(
                segmentation[reference]['raw_indices'],
                reduce(union1d, superior_indices, -1))

        # Log a message about the overlap removal
        get_logger().info("Removing segment overlaps ...")

        # Get the segmentation data
        segmentation = self.segmentation

        # Remove the overlaps from all segments
        apply(remove_segment_overlap, (*segmentation,))

    def resize_segments(
            self,
            shape):
        """
        Resize the segments from the CT array shape to a given shape.

        Parameters
        ----------
        shape : list, ndarray or tuple
            Shape of the resized array.
        """

        def resize_segment(segment):
            """Resize a segment."""

            # Initialize the segment mask
            mask = zeros(ct_shape)

            # Fill the mask at the segment indices
            mask[unravel_index(
                segmentation[segment]['prioritized_indices'], ct_shape,
                order='F')] = 1

            # Get the zoom factors for all cube dimensions
            zooms = (pair[0]/pair[1] for pair in zip(shape, ct_shape))

            # Store the resized indices
            segmentation[segment]['resized_indices'] = ravel_multi_index(
                where(zoom(mask, zooms, order=0)), shape, order='F')

        # Get the CT shape and segmentation data
        ct_shape, segmentation = (
            self.computed_tomography['cube_dimensions'], self.segmentation)

        # Log a message about the segment resizing
        get_logger().info(
            "Resizing segments from CT array shape %s to shape %s ...",
            tuple(ct_shape.tolist()), tuple(shape))

        # Resize all segments
        apply(resize_segment, (*segmentation,))
