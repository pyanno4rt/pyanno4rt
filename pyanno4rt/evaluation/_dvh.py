"""DVH computation."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from numpy import array, linspace, logical_and, unravel_index

# %% Internal package import

from pyanno4rt.datahub import Datahub

# %% Class definition


class DVH():
    """
    DVH computation class.

    This class provides methods to compute the dose-volume histogram (DVH) as \
    a means to evaluate dose distributions from treatment plans across all \
    segments. Both cumulative and differential DVH can be obtained.

    Parameters
    ----------
    dvh_type : {'cumulative', 'differential'}
        Type of DVH to be calculated.

    number_of_points : int
        Number of (evenly-spaced) points for which to evaluate the DVH.

    display_segments : list
        Names of the segmented structures to be displayed.

    Attributes
    ----------
    dvh_type : {'cumulative', 'differential'}
        See 'Parameters'.

    number_of_points : int
        See 'Parameters'.

    display_segments : list
        See 'Parameters'.
    """

    def __init__(
            self,
            dvh_type,
            number_of_points,
            display_segments):

        # Initialize the datahub
        hub = Datahub()

        # Log a message about the initialization of the class
        hub.logger.display_info("Initializing DVH class ...")

        # Get the instance attributes from the arguments
        self.dvh_type = dvh_type
        self.number_of_points = number_of_points
        self.display_segments = (display_segments
                                 if len(display_segments) > 0
                                 else list(hub.segmentation.keys()))

    def compute(
            self,
            dose_cube):
        """
        Compute the cumulative or differential DVH for all segments.

        Parameters
        ----------
        dose_cube : ndarray
            Three-dimensional array with the dose values (on the CT grid).
        """

        # Initialize the datahub
        hub = Datahub()

        # Log a message about the DVH computation
        hub.logger.display_info(f"Computing {self.dvh_type} DVH with "
                                f"{self.number_of_points} points for all "
                                "segments ...")

        def get_evaluation_points():
            """Get the points at which to evaluate the DVH."""

            # Get the minimum and the maximum dose from the dose cube
            min_dose, max_dose = dose_cube.min(), dose_cube.max()

            # Map the DVH type to the dose intervals
            bounds = {'cumulative': (0, 1.05*max_dose),
                      'differential': (0.95*min_dose, 1.05*max_dose)}

            return linspace(*bounds[self.dvh_type], self.number_of_points,
                            endpoint=True)

        def compute_cumulative_dvh(dose, evaluation_points):
            """Compute the cumulative DVH points."""

            return array([(dose >= point).sum()
                          for point in evaluation_points])

        def compute_differential_dvh(dose, evaluation_points):
            """Compute the differential DVH points."""

            # Determine the bin radius
            bin_radius = (evaluation_points[1] - evaluation_points[0])/2

            return array([sum(logical_and(point - bin_radius < dose,
                                          point + bin_radius > dose))
                          for point in evaluation_points])

        def get_segment_dvh(indices, cube_dimensions, evaluation_points):
            """Get the DVH for a single segment."""

            # Get the dose values for the segment
            segment_dose = dose_cube[
                unravel_index(indices, cube_dimensions, order='F')]

            return 100*(dvh_functions[self.dvh_type](
                segment_dose, evaluation_points) / len(indices))

        # Map the DVH type to the computation function
        dvh_functions = {'cumulative': compute_cumulative_dvh,
                         'differential': compute_differential_dvh}

        # Initialize the histogram dictionary with the evaluation points
        histogram = {'evaluation_points': get_evaluation_points()}

        # Add the segment names with the corresponding DVH values
        histogram.update({segment: {'dvh_values': get_segment_dvh(
            hub.segmentation[segment]['raw_indices'],
            hub.computed_tomography['cube_dimensions'],
            histogram['evaluation_points'])}
            for segment in (*hub.segmentation,)})

        # Add the segment names to be displayed
        histogram['display_segments'] = self.display_segments

        # Enter the histogram dictionary into the datahub
        hub.histogram = histogram
