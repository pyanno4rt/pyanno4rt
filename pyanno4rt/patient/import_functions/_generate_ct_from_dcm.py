"""DICOM folder-based CT dictionary generation."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from numpy import array, clip, dstack, prod
from pydicom.pixel_data_handlers.util import apply_modality_lut
from scipy.interpolate import interp1d
from scipy.ndimage import zoom

# %% Internal package import

from pyanno4rt.tools import arange_with_endpoint

# %% Function definition


def generate_ct_from_dcm(data, resolution):
    """
    Generate the CT dictionary from a folder with DICOM (.dcm) files.

    Parameters
    ----------
    data : tuple
        Tuple of :class:`pydicom.dataset.FileDataset` objects with \
        information on the CT slices.

    resolution : None or list
        Imaging resolution for post-processing interpolation of the CT and \
        segmentation data.

    Returns
    -------
    computed_tomography : dict
        Dictionary with information on the CT images.

    Raises
    ------
    ValueError
        If either the grid resolutions, the image positions or the \
        dimensionalities are inconsistent.
    """

    # Specify the Hounsfield lookup table (HU to RED/RSP)
    hlut = ((-1024.0, 200.0, 449.0, 2000.0, 2048.0, 3071.0),
            (0.00324, 1.2, 1.20001, 2.49066, 2.5306, 2.53061))

    def check_ct_data(data):
        """Check the CT data from the DICOM files."""

        # Check if the grid resolutions are inconsistent
        if any(len(set(resolutions)) != 1 for resolutions in zip(
                *((file.PixelSpacing[1],
                   file.PixelSpacing[0],
                   file.SliceThickness)
                  for file in data))):

            # Raise an error to indicate an inconsistency
            raise ValueError(
                "The grid resolution is found to be inconsistent across "
                "the CT slices!")

        # Check if the image positions are inconsistent
        if any(len(set(positions)) != 1 for positions in zip(
                *((file.ImagePositionPatient[1], file.ImagePositionPatient[0])
                  for file in data))):

            # Raise an error to indicate an inconsistency
            raise ValueError(
                "The imaging position of the patient is found to be "
                "inconsistent across the CT slices!")

        # Check if the dimensionalities are inconsistent
        if any(len(set(dimensions)) != 1 for dimensions in zip(
                *((file.Columns, file.Rows) for file in data))):

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

    def interpolate_ct_dictionary(computed_tomography, resolution):
        """Interpolate the CT dictionary values to a resolution."""

        # Get the current cube dimensions
        old_dimensions = computed_tomography['cube_dimensions']

        # Update the grid resolution
        computed_tomography['resolution'] = dict(
            zip(('x', 'y', 'z'), resolution))

        # Loop over the grid axes
        for index, axis in enumerate(('x', 'y', 'z')):

            # Update the grid points on the current axis
            computed_tomography[axis] = arange_with_endpoint(
                computed_tomography[axis][0],
                computed_tomography[axis][-1],
                resolution[index])

        # Update the cube dimensions
        computed_tomography['cube_dimensions'] = array([
            len(computed_tomography[axis]) for axis in ('x', 'y', 'z')])

        # Get the zoom factors for all cube dimensions
        zooms = (pair[0]/pair[1] for pair in zip(
            computed_tomography['cube_dimensions'], old_dimensions))

        # Interpolate the CT cube to the target resolution
        computed_tomography['cube'] = zoom(
            computed_tomography['cube'], zooms, order=1)

        # Update the number of voxels
        computed_tomography['number_of_voxels'] = prod(
            computed_tomography['cube_dimensions'])

        return computed_tomography

    # Check the CT data
    check_ct_data(data)

    # Initialize the dictionary
    computed_tomography = {}

    # Add the interpolated RED/RSP cube to the dictionary
    computed_tomography['cube'] = calculate_3d_cube(data)

    # Add the grid resolution to the dictionary
    computed_tomography['resolution'] = {
        'x': data[0].PixelSpacing[1],
        'y': data[0].PixelSpacing[0],
        'z': data[0].SliceThickness}

    # Add the grid points in x to the dictionary
    computed_tomography['x'] = array([
        data[0].ImagePositionPatient[0] + factor*data[0].PixelSpacing[1]
        for factor in range(data[0].Columns)])

    # Add the grid points in y to the dictionary
    computed_tomography['y'] = array([
        data[0].ImagePositionPatient[1] + factor*data[0].PixelSpacing[0]
        for factor in range(data[0].Rows)])

    # Add the grid points in z to the dictionary
    computed_tomography['z'] = array([
        file.ImagePositionPatient[2] for file in data])

    # Add the cube dimensions to the dictionary
    computed_tomography['cube_dimensions'] = array(
        computed_tomography['cube'].shape)

    # Add the number of voxels to the dictionary
    computed_tomography['number_of_voxels'] = prod(
        computed_tomography['cube_dimensions'])

    # Check if a target resolution has been passed
    if resolution:

        # Return the interpolated CT dictionary
        return interpolate_ct_dictionary(computed_tomography, resolution)

    return computed_tomography
