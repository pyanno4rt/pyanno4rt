"""DVH."""

# Author: Tim Ortkamp

# %% External package import

from numba import njit
from numpy import array, linspace, logical_and, nan, unravel_index

# %% Internal package import

from pyanno4rt.logging import get_logger

# %% Class definition


class DVH():
    """
    DVH class.

    This class provides methods to evaluate dose distributions via \
    cumulative or differential dose-volume histograms (DVHs).

    Parameters
    ----------
    dvh_type : {'cumulative', 'differential'}
        Type of DVH to be evaluated.

    number_of_points : int
        Number of (evenly-spaced) DVH evaluation points.

    Attributes
    ----------
    dvh_type : {'cumulative', 'differential'}
        See 'Parameters'.

    number_of_points : int
        See 'Parameters'.

    histogram : dict
        Dictionary with information on the DVHs.
    """

    def __init__(
            self,
            dvh_type,
            number_of_points):

        # Log a message about the initialization of the class
        get_logger().info("Initializing DVH ...")

        # Get the input attributes
        self.dvh_type = dvh_type
        self.number_of_points = number_of_points

        # Initialize the histogram dictionary
        self.histogram = None

    def even_points(
            self,
            dose):
        """
        Get the evenly spaced DVH evaluation points.

        Parameters
        ----------
        dose : ndarray
            Array with the dose values.

        Returns
        -------
        ndarray
            Array with the evenly spaced DVH evaluation points.
        """

        # Get the minimum/maximum dose
        minimum_dose, maximum_dose = dose.min(), dose.max()

        # Map the DVH types to the dose intervals
        intervals = {
            'cumulative': (0, 1.05*maximum_dose),
            'differential': (0.95*minimum_dose, 1.05*maximum_dose)}

        # Return the evenly spaced evaluation points
        return linspace(
            *intervals[self.dvh_type], self.number_of_points, endpoint=True)

    def evaluate_array(
            self,
            dose,
            points=None):
        """
        Evaluate the DVH for a single dose array.

        Parameters
        ----------
        dose : ndarray
            Array with the dose values.

        points : ndarray, default=None
            Array with the DVH evaluation points. If None, points are \
            calculated with :meth:`~pyanno4rt.evaluation._dvh.DVH.even_points`.

        Returns
        -------
        ndarray
            Array with the DVH values.
        """

        # Check if any dose values have been passed
        if len(dose) > 0:

            # Check if no evaluation points have been passed
            if points is None:

                # Get the evenly spaced DVH evaluation points
                evaluation_points = self.even_points(dose)

            else:

                # Use the input DVH evaluation points
                evaluation_points = points

            # Map the DVH types to the calculation functions
            dvh_functions = {
                'cumulative': cumulate, 'differential': differentiate}

            # Return the DVH values
            return dvh_functions[self.dvh_type](dose, evaluation_points)

        # Return NaNs
        return array([nan]*self.number_of_points)

    def evaluate_segments(
            self,
            segmentation,
            dose_cube):
        """
        Evaluate the DVH for all segments.

        Parameters
        ----------
        segmentation : dict
            Dictionary with information on the segments.

        dose_cube : ndarray
            Array with the dose values.
        """

        # Log a message about the DVH evaluation
        get_logger().info(
            "Evaluating %s DVH with %s points for all segments ...",
            self.dvh_type, self.number_of_points)

        # Get the evenly spaced DVH evaluation points
        evaluation_points = self.even_points(dose_cube)

        # Get the dose per segment
        doses = {segment: dose_cube[unravel_index(
            segmentation[segment]['raw_indices'], dose_cube.shape, order='F')]
            for segment in segmentation}

        # Calculate the DVH values
        values = {
            segment: self.evaluate_array(doses[segment], evaluation_points)
            for segment in segmentation}

        # Get the histogram dictionary
        self.histogram = {'evaluation_points': evaluation_points} | values


@njit
def cumulate(dose, points):
    """
    Evaluate the cumulative DVH.

    Parameters
    ----------
    dose : ndarray
        Array with the dose values.

    points : ndarray
        Array with the DVH evaluation points.

    Returns
    -------
    ndarray
        Array with the DVH values.
    """

    # Return the DVH values
    return array([(dose >= point).sum() for point in points]) / len(dose)


@njit
def differentiate(dose, points):
    """
    Evaluate the differential DVH.

    Parameters
    ----------
    dose : ndarray
        Array with the dose values.

    points : ndarray
        Array with the DVH evaluation points.

    Returns
    -------
    ndarray
        Array with the DVH values.
    """

    # Determine the bin radius
    radius = (points[1] - points[0]) / 2

    # Return the DVH values
    return array([
        sum(logical_and(point - radius < dose, point + radius > dose))
        for point in points]) / len(dose)
