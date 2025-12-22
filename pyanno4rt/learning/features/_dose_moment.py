"""Dose moment feature."""

# Author: Tim Ortkamp

# %% External package import

from jax import grad, jit
from jax.numpy import array, int32, meshgrid
from jax.numpy import sum as jsum

# %% Internal package import

from pyanno4rt.learning.features import DosiomicFeature

# %% Class definition


class DoseMoment(DosiomicFeature):
    """Dose moment feature class."""

    @staticmethod
    def value(
            coefficients,
            _,
            *args):
        """
        Compute the dose moment.

        Parameters
        ----------
        coefficients : str
            Moment coefficient string, e.g. '111'.

        _ : ndarray
            Dose array.

        *args : tuple
            Optional (non-keyworded) parameters. args[0] should be the dose \
            cube, args[1] the resolution, and args[2] the binary segment mask.

        Returns
        -------
        object of class :class:`~jaxlib.xla_extension.ArrayImpl`
            Dose moment value.
        """

        def compute_scaled_cube(points):
            """Compute the mean of the scaled dose cube."""

            return jsum((points*dose_cube)) / jsum(dose_cube)

        # Get the dose cube from the argument
        dose_cube = args[0]

        # Get the coefficients of the moment function from the argument
        coeff_1, coeff_2, coeff_3 = tuple(map(int32, coefficients))

        # Determine the axis points from a meshed grid
        points_x, points_y, points_z = meshgrid(
            array(range(dose_cube.shape[0])),
            array(range(dose_cube.shape[1])),
            array(range(dose_cube.shape[2])))

        # Compute the means of the axis points
        mean_x, mean_y, mean_z = tuple(
            map(compute_scaled_cube, (points_x, points_y, points_z)))

        # Compute the moment function numerator
        numerator = jsum(
            (points_x-mean_x)**coeff_1 * (points_y-mean_y)**coeff_2
            * (points_z-mean_z)**coeff_3 * dose_cube)

        # Compute the moment function denominator
        denominator = jsum(dose_cube)**((coeff_1+coeff_2+coeff_3)/3 + 1)

        return numerator/denominator

    @staticmethod
    def compute(
            coefficients,
            dose,
            *args):
        """
        Check the jitting status and call the value function.

        Parameters
        ----------
        coefficients : str
            Moment coefficient string, e.g. '111'.

        dose : ndarray
            Dose array.

        *args : tuple
            Optional (non-keyworded) parameters. args[0] should be the dose \
            cube, args[1] the resolution, and args[2] the binary segment mask.

        Returns
        -------
        object of class :class:`~jaxlib.xla_extension.ArrayImpl`
            Dose moment value.
        """

        # Check if the value function has not yet been jitted
        if not DoseMoment.value_is_jitted:

            # Perform the jitting
            DoseMoment.value_function = jit(DoseMoment.value, static_argnums=0)

            # Set 'value_is_jitted' to True
            DoseMoment.value_is_jitted = True

        return DoseMoment.value_function(coefficients, dose, *args)

    @staticmethod
    def differentiate(
            coefficients,
            dose,
            *args):
        """
        Check the jitting status and call the gradient function.

        Parameters
        ----------
        coefficients : str
            Moment coefficient string, e.g. '111'.

        dose : ndarray
            Dose array.

        *args : tuple
            Optional (non-keyworded) parameters. args[0] should be the dose \
            cube, args[1] the resolution, and args[2] the binary segment mask.

        Returns
        -------
        object of class :class:`~jaxlib.xla_extension.ArrayImpl`
            Dose moment gradient.
        """

        # Check if the gradient function has not yet been jitted
        if not DoseMoment.gradient_is_jitted:

            # Perform the jitting
            DoseMoment.gradient_function = jit(grad(
                DoseMoment.value, argnums=2), static_argnums=0)

            # Set 'gradient_is_jitted' to True
            DoseMoment.gradient_is_jitted = True

        return DoseMoment.gradient_function(
            coefficients, dose, *args).reshape(-1)
