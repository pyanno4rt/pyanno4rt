"""Dosimetrics."""

# Author: Tim Ortkamp

# %% External package import

from statistics import mean

from numpy import array, floor, linspace, sort, unravel_index
from scipy.interpolate import interp1d

# %% Internal package import

from pyanno4rt.logging import get_logger
from pyanno4rt.tools import get_objectives

# %% Class definition


class Dosimetrics():
    """
    Dosimetrics class.

    This class provides methods to evaluate dose distributions via metrics, \
    including statistical location and dispersion measures, DVH indicators as \
    well as conformity (CI) and homogeneity index (HI).

    Parameters
    ----------
    reference_volumes : list
        Reference volumes for which to evaluate the inverse DVH values.

    reference_doses : list
        Reference dose values for which to evaluate the DVH values.

    number_of_fractions : int
        Number of fractions according to the treatment scheme.

    Attributes
    ----------
    reference_volumes : tuple
        See 'Parameters'.

    reference_doses : tuple
        See 'Parameters'.

    number_of_fractions : int
        See 'Parameters'.

    quantities : dict
        Dictionary with information on the dosimetrics.
    """

    def __init__(
            self,
            reference_volumes,
            reference_doses,
            number_of_fractions):

        # Log a message about the initialization of the class
        get_logger().info("Initializing dosimetrics ...")

        # Get the input attributes
        self.reference_volumes = tuple(sorted(reference_volumes))
        self.reference_doses = tuple(sorted(reference_doses))
        self.number_of_fractions = number_of_fractions

        # Initialize the quantities dictionary
        self.quantities = {}

    def evaluate_array(
            self,
            dose,
            dose_cube=None,
            prescription=None):
        """
        Evaluate the dosimetrics for a single dose array.

        Parameters
        ----------
        dose : ndarray
            Array with the dose values.

        dose_cube : ndarray, default=None
            Array with the "full body" dose values.

        prescription : int or float, default=None
            Dose prescription value.

        Returns
        -------
        dict
            Dictionary with information on the dosimetrics.
        """

        # Check if the reference doses should be calculated
        if not self.reference_doses and dose_cube is not None:

            # Update the reference doses
            self.reference_doses = tuple(
                floor(linspace(0, dose_cube.max(), 5)*10)/10)

        # Initialize the quantities dictionary
        quantities = {}

        # Compute the base statistics
        quantities['D_mean'] = self.mean(dose)
        quantities['D_std'] = self.deviation(dose)
        quantities['D_min'] = self.minimum(dose)
        quantities['D_max'] = self.maximum(dose)

        # Initialize the dose interpolator
        interpolator = interp1d(
            linspace(0, 1, len(dose)), dose, fill_value='extrapolate')

        # Compute the inverse DVH values
        quantities |= dict(zip(
            (f'D_{level}' for level in self.reference_volumes),
            self.dx(dose, self.reference_volumes, interpolator)))

        # Compute the DVH values
        quantities |= dict(zip(
            (f'V_{level}' for level in self.reference_doses),
            self.vx(dose, self.reference_doses)))

        # Check if a prescription value has been provided
        if prescription is not None:

            # Check if a dose cube has been provided
            if dose_cube is not None:

                # Compute the conformity index
                quantities['CI'] = (
                    self.conformity_index(dose, dose_cube, prescription))

            # Compute the homogeneity index
            quantities['HI'] = (
                self.homogeneity_index(dose, prescription, interpolator))

        return quantities

    def evaluate_segments(
            self,
            segmentation,
            components,
            dose_cube):
        """
        Evaluate the dosimetrics for all segments.

        Parameters
        ----------
        segmentation : dict
            Dictionary with information on the segments.

        components : list
            Plan components.

        dose_cube : ndarray
            Array with the "full body" dose values.
        """

        # Log a message about the dosimetrics evaluation
        get_logger().info("Evaluating dosimetrics for all segments ...")

        # Loop over the segments
        for segment, data in segmentation.items():

            # Get the sorted dose vector
            dose = sort(dose_cube[unravel_index(
                data['raw_indices'], dose_cube.shape, order='F')])

            # Get the wrapped objective
            objective = [
                component for component in get_objectives(components)
                if component.name in (
                        'Squared Deviation', 'Squared Underdosing')
                and segment in component.segment]

            # Filter the objective by component name
            objective = tuple(filter(lambda item: item.name in (
                'Squared Deviation', 'Squared Underdosing'), objective))

            # Initialize the dose prescription
            prescription = None

            # Check if the segment is a target of interest
            if data['type'] == 'TARGET' and len(objective) > 0:

                # Get the mean dose prescription
                prescription = self.number_of_fractions*mean(
                    item.parameter_value[0] for item in objective)

            # Get the segment dosimetrics
            self.quantities[segment] = self.evaluate_array(
                dose, dose_cube, prescription)

    def mean(
            self,
            dose):
        """
        Calculate the mean dose.

        Parameters
        ----------
        dose : ndarray
            Array with the dose values.

        Returns
        -------
        float
            Mean dose value.
        """

        return dose.mean()

    def deviation(
            self,
            dose):
        """
        Calculate the dose standard deviation.

        Parameters
        ----------
        dose : ndarray
            Array with the dose values.

        Returns
        -------
        float
            Dose standard deviation value.
        """

        return dose.std()

    def minimum(
            self,
            dose):
        """
        Calculate the minimum dose.

        Parameters
        ----------
        dose : ndarray
            Array with the dose values.

        Returns
        -------
        float
            Minimum dose value.
        """

        return dose.min()

    def maximum(
            self,
            dose):
        """
        Calculate the maximum dose.

        Parameters
        ----------
        dose : ndarray
            Array with the dose values.

        Returns
        -------
        float
            Maximum dose value.
        """

        return dose.max()

    def dx(
            self,
            dose,
            levels=(2, 5, 50, 95, 98),
            interpolator=None):
        """
        Calculate the inverse DVH value(s).

        Parameters
        ----------
        dose : ndarray
            Array with the dose values.

        levels : tuple or list, default=(2, 5, 50, 95, 98)
            Relative volumes for which to evaluate the inverse DVH value(s).

        interpolator : object of class \
            :class:`~scipy.interpolate._interpolate.interp1d`, default=None
            The object used to interpolate between dose values.

        Returns
        -------
        ndarray
            Inverse DVH (Dx) value(s).
        """

        # Check if no interpolator has been passed
        if interpolator is None:

            # Initialize the dose interpolator
            interpolator = interp1d(
                linspace(0, 1, len(dose)), dose, fill_value='extrapolate')

        return array([interpolator(1-level/100) for level in levels])

    def vx(
            self,
            dose,
            levels=()):
        """
        Calculate the DVH value(s).

        Parameters
        ----------
        dose : ndarray
            Array with the dose values.

        levels : tuple or list, default=()
            Dose values for which to evaluate the DVH value(s).

        Returns
        -------
        ndarray
            DVH (Vx) value(s).
        """

        # Check if no dose levels have been passed
        if len(levels) == 0:

            # Get the default dose levels
            levels = floor(linspace(0, dose.max(), 5)*10)/10

        return array([(dose >= level).sum()/len(dose) for level in levels])

    def conformity_index(
            self,
            dose,
            dose_cube,
            prescription,
            isolevel=95):
        """
        Calculate the dose conformity index.

        Parameters
        ----------
        dose : ndarray
            Array with the target dose values.

        dose_cube : ndarray
            Array with the "full body" dose values.

        prescription : int or float
            Dose prescription value.

        isolevel : int or float, default=95
           Isodose level.

        Returns
        -------
        float
            Dose conformity index.
        """

        # Get the isodose
        isodose = prescription*isolevel/100

        return (
            ((dose >= isodose).sum())**2
            / (len(dose)*(dose_cube >= isodose).sum()))

    def homogeneity_index(
            self,
            dose,
            prescription,
            interpolator=None):
        """
        Calculate the dose homogeneity index.

        Parameters
        ----------
        dose : ndarray
            Array with the target dose values.

        prescription : int or float
            Dose prescription value.

        interpolator : object of class \
            :class:`~scipy.interpolate._interpolate.interp1d`, default=None
            The object used to interpolate between dose values.

        Returns
        -------
        float
            Dose homogeneity index.
        """

        # Check if no interpolator has been passed
        if interpolator is None:

            # Initialize the dose interpolator
            interpolator = interp1d(
                linspace(0, 1, len(dose)), dose, fill_value='extrapolate')

        return 100*(interpolator(0.95)-interpolator(0.05))/prescription
