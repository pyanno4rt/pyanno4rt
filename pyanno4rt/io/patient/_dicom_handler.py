"""DICOM file handler."""

# Author: Tim Ortkamp

# %% External package import

from os import listdir

from colorsys import hsv_to_rgb
from numpy import (
    array, clip, column_stack, dstack, logical_or, ravel_multi_index, prod,
    sort, where, zeros)
from pydicom import dcmread
from pydicom.pixels import apply_modality_lut
from scipy.interpolate import interp1d
from skimage.draw import polygon2mask

# %% Internal package import

from pyanno4rt.logging import get_logger

# %% Class definition


class DicomHandler():
    """
    DICOM file handler class.

    This class provides methods to handle patient imaging data from DICOM \
    files and generate the CT and segmentation dictionaries.

    Notes
    -----
    DICOM file export is not implemented yet. This will be considered in a \
    later release.
    """

    def __init__(self):

        pass

    def load(
            self,
            path):
        """
        Load the patient imaging data.

        Parameters
        ----------
        path : str
            Path to the patient imaging data.

        Returns
        -------
        dict
            Dictionary with information on the CT images.

        dict
            Dictionary with information on the segments.
        """

        # Load the DICOM files
        files = tuple(dcmread(f'{path}/{file}') for file in listdir(path))

        # Get the (axially ordered) CT data files
        ct_dicom = tuple(sorted(
            (file for file in files if hasattr(file, 'PixelData')),
            key=lambda file: file.ImagePositionPatient[2]))

        # Get the segmentation data file
        segmentation_dicom = next(
            file for file in files if hasattr(file, 'ROIContourSequence'))

        # Generate the CT dictionary
        computed_tomography = self.generate_ct(ct_dicom)

        # Generate the segmentation dictionary
        segmentation = self.generate_segmentation(
            segmentation_dicom, ct_dicom, computed_tomography)

        return computed_tomography, segmentation

    def save(
            self,
            computed_tomography,
            segmentation,
            path):
        """
        Save the patient imaging data.

        Parameters
        ----------
        computed_tomography : dict
            Dictionary with information on the CT images.

        segmentation : dict
            Dictionary with information on the segments.

        path : str
            Path for storing the patient imaging data.
        """

        raise NotImplementedError(
            "DICOM file export is not implemented yet - you may select "
            "another file format (.mat)!")

    def generate_ct(
            self,
            data):
        """
        Generate the CT dictionary.

        Parameters
        ----------
        data : tuple
            Tuple of :class:`pydicom.dataset.FileDataset` objects with \
            information on the CT slices.

        Returns
        -------
        dict
            Dictionary with information on the CT images.

        Raises
        ------
        ValueError
            If either the grid resolutions, the image positions or the \
            dimensionalities are inconsistent.
        """

        # Specify the Hounsfield lookup table (HU to RED/RSP)
        hlut = (
            (-1024.0, 200.0, 449.0, 2000.0, 2048.0, 3071.0),
            (0.00324, 1.2, 1.20001, 2.49066, 2.5306, 2.53061))

        def validate_ct_data(data):
            """Validate the CT data."""

            # Check if the grid resolutions are inconsistent
            if any(len(set(resolutions)) != 1 for resolutions in zip(*((
                    file.PixelSpacing[1], file.PixelSpacing[0],
                    file.SliceThickness) for file in data))):

                # Raise an error to indicate an inconsistency
                raise ValueError(
                    "The grid resolution is found to be inconsistent across "
                    "the CT slices!")

            # Check if the image positions are inconsistent
            if any(len(set(positions)) != 1 for positions in zip(*((
                    file.ImagePositionPatient[1], file.ImagePositionPatient[0])
                  for file in data))):

                # Raise an error to indicate an inconsistency
                raise ValueError(
                    "The imaging position of the patient is found to be "
                    "inconsistent across the CT slices!")

            # Check if the dimensionalities are inconsistent
            if any(len(set(dimensions)) != 1 for dimensions in zip(*((
                    file.Columns, file.Rows) for file in data))):

                # Raise an error to indicate an inconsistency
                raise ValueError(
                    "The number of data columns or rows is found to be "
                    "inconsistent across the CT slices!")

        def calculate_3d_cube(data):
            """Calculate the CT cube from the pixel arrays."""

            # Generate the 3D cube with HU values
            cube_hounsfield = dstack(tuple(
                apply_modality_lut(file.pixel_array, file) for file in data))

            # Clip the HU values before interpolation
            clip(cube_hounsfield, a_min=cube_hounsfield.min(),
                 a_max=cube_hounsfield.max(), out=cube_hounsfield)

            # Initialize the interpolator
            interpolator = interp1d(hlut[0], hlut[1], 'linear')

            return interpolator(cube_hounsfield)

        # Validate the CT data
        validate_ct_data(data)

        # Initialize the dictionary
        computed_tomography = {}

        # Add the interpolated RED/RSP cube
        computed_tomography['cubeHU'] = calculate_3d_cube(data)

        # Add the grid resolution
        computed_tomography['resolution'] = {
            'x': data[0].PixelSpacing[0],
            'y': data[0].PixelSpacing[1],
            'z': data[0].SliceThickness}

        # Add the grid points in x
        computed_tomography['x'] = array([
            data[0].ImagePositionPatient[0] + factor*data[0].PixelSpacing[0]
            for factor in range(data[0].Columns)])

        # Add the grid points in y
        computed_tomography['y'] = array([
            data[0].ImagePositionPatient[1] + factor*data[0].PixelSpacing[1]
            for factor in range(data[0].Rows)])

        # Add the grid points in z
        computed_tomography['z'] = array([
            file.ImagePositionPatient[2] for file in data])

        # Add the cube dimensions
        computed_tomography['cube_dimensions'] = array(
            computed_tomography['cubeHU'].shape)

        # Add the number of voxels
        computed_tomography['number_of_voxels'] = prod(
            computed_tomography['cube_dimensions'])

        return computed_tomography

    def generate_segmentation(
            self,
            data,
            ct_dicom,
            ct_dictionary):
        """
        Generate the segmentation dictionary.

        Parameters
        ----------
        data : object of class :class:`pydicom.dataset.FileDataset`
            The object used to represent the segmentation data.

        ct_dicom : tuple
            Tuple of :class:`pydicom.dataset.FileDataset` objects with \
            information on the CT slices.

        ct_dictionary : dict
            Dictionary with information on the CT images.

        Returns
        -------
        dict
            Dictionary with information on the segments.

        Raises
        ------
        ValueError
            If the contour sequence for a segment includes out-of-slice points.
        """

        def generate_colors(length):
            """Generate a tuple of certain length with different RGB colors."""

            return tuple(
                array(hsv_to_rgb(value/(length+1), 1.0, 1.0))
                for value in range(length))

        def compute_segment_indices(
                ct_slices, computed_tomography, roi_contour):
            """Compute the (sorted) binary segment indices."""

            # Initialize the segment cube
            segment_cube = zeros(computed_tomography['cube_dimensions'])

            # Loop over the contour sequences
            for sequence in roi_contour.ContourSequence:

                # Check if the geometric type is different from 'POINT'
                if sequence.ContourGeometricType != 'POINT':

                    # Get the grid points of the sequence
                    points_x, points_y, points_z = (
                        sequence.ContourData[i::3] for i in range(3))

                    # Loop over the grid point dimensions
                    for points in (points_x, points_y, points_z):

                        # Close the contour polygon by adding the first point
                        points.append(points[0])

                    # Round the z-points to account for numerical issues
                    points_z = [1e-10*round(1e10*value) for value in points_z]

                    # Check if contour points outside the slice exist
                    if len(set(points_z)) > 1:

                        # Log a message about the out-of-slice points
                        logger.error(
                            "The contour sequence for the segment %s includes "
                            "out-of-slice points!", segment)

                        # Raise an error to indicate out-of-slice points
                        raise ValueError(
                            f"The contour sequence for the segment {segment} "
                            "includes out-of-slice points!")

                    # Check if the current contour slice lies within the data
                    if (min(computed_tomography['z'])
                            <= points_z[0]
                            <= max(computed_tomography['z'])):

                        # Interpolate the points on the x- and y-axis
                        interpolated_x, interpolated_y = (interp1d(
                            computed_tomography[axis[0]],
                            range(computed_tomography[
                                'cube_dimensions'][axis[1]]),
                            'linear', fill_value='extrapolate')(axis[2])
                            for axis in (
                                ('x', 1, points_x), ('y', 0, points_y)))

                        # Convert the polygon vertices into a binary mask
                        mask = polygon2mask(
                            computed_tomography['cube_dimensions'][:2],
                            column_stack((interpolated_y, interpolated_x)))

                        # Get the CT slice indices
                        ct_slice_indices = [
                            index for index, value in enumerate(
                                computed_tomography['z'])
                            if (points_z[0]
                                - int(ct_slices[0].SliceThickness)/2
                                <= value <
                                points_z[0]
                                + int(ct_slices[0].SliceThickness)/2)]

                        # Loop over the slice indices
                        for index in ct_slice_indices:

                            # Enter the binary mask into the segment cube
                            segment_cube[:, :, index] = logical_or(
                                segment_cube[:, :, index], mask)

                    else:

                        # Log a message about the missing CT data
                        logger.warning(
                            "Omitting contour data for '%s' at slice position "
                            "%s mm - no CT data available ...",
                            segment, points_z[0])

            return sort(ravel_multi_index(
                where(segment_cube), segment_cube.shape, order='F'))

        # Initialize the logger
        logger = get_logger()

        # Get the default color tuple
        default_colors = generate_colors(len(data.ROIContourSequence))

        # Initialize the segmentation dictionary
        segmentation = {}

        # Loop over the ROI contours
        for roi_contour in data.ROIContourSequence:

            # Find the corresponding ROI structure from the index number
            roi_structure = next(
                sequence for sequence in data.StructureSetROISequence
                if roi_contour.ReferencedROINumber == sequence.ROINumber)

            # Get the structure name
            segment = roi_structure.ROIName

            # Add the first-layer backbone
            segmentation[segment] = {
                key: None for key in (
                    'index', 'type', 'raw_indices', 'parameters', 'objective',
                    'constraint')}

            # Add the second-layer backbone
            segmentation[segment]['parameters'] = {
                key: None for key in (
                    'priority', 'alphaX', 'betaX', 'visibleColor')}

            # Add the segment index
            segmentation[segment]['index'] = (
                int(roi_contour.ReferencedROINumber)-1)

            # Check if the segment is a target volume
            if any(string in segment.lower() for string in (
                    'tv', 'target', 'gtv', 'ctv', 'ptv', 'boost', 'tumor')):

                # Add the 'TARGET' type
                segmentation[segment]['type'] = 'TARGET'

                # Add the default target priority
                segmentation[segment]['parameters']['priority'] = 1

            else:

                # Add the 'OAR' type
                segmentation[segment]['type'] = 'OAR'

                # Add the default organ-at-risk priority
                segmentation[segment]['parameters']['priority'] = 2

            # Add the default biological parameters
            segmentation[segment]['parameters']['alphaX'] = 0.1
            segmentation[segment]['parameters']['betaX'] = 0.05

            # Check if the ROI contour includes a display color
            if hasattr(roi_contour, 'ROIDisplayColor'):

                # Add the visible color
                segmentation[segment]['parameters']['visibleColor'] = array(
                    [int(x)/255 for x in roi_contour.ROIDisplayColor])

            else:

                # Add the default visible color
                segmentation[segment]['parameters']['visibleColor'] = (
                    default_colors[segmentation[segment]['index']])

            # Check if the ROI contour includes contour sequence data
            if (hasattr(roi_contour, 'ContourSequence')
                    and roi_contour.ContourSequence):

                # Add the segment indices
                segmentation[segment]['raw_indices'] = compute_segment_indices(
                    ct_dicom, ct_dictionary, roi_contour)

            else:

                # Log a message about the empty ROI contour
                logger.warning(
                    "ROI contour for '%s' is empty ...", segment)

    	# Filter the segments
        segmentation = {
            key: value for key, value in segmentation.items()
            if len(value['raw_indices']) > 0}

        return dict(sorted(segmentation.items()))
