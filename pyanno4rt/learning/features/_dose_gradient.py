"""Dose gradient feature."""

# Author: Tim Ortkamp

# %% External package import

from jax import grad, jit
from jax.numpy import count_nonzero, gradient
from jax.numpy import sum as jsum

# %% Internal package import

from pyanno4rt.learning.features import DosiomicFeature

# %% Class definition


class DoseGradient(DosiomicFeature):
    """Dose gradient feature class."""

    @staticmethod
    def value(*args):
        """
        Compute the dose gradient.

        Parameters
        ----------
        *args : tuple
            Optional (non-keyworded) parameters. args[0] should be the dose \
            cube, args[1] the resolution, args[2] the binary segment mask, \
            and args[3] the axis label, e.g. 'x'.

        Returns
        -------
        object of class :class:`~jaxlib.xla_extension.ArrayImpl`
            Dose gradient value.
        """

        # Map the axes to the argument values
        axis_to_args = {
            'x': [args[1][0], 1],
            'y': [args[1][1], 0],
            'z': [args[1][2], 2]}

        # Extract the arguments
        resolution, grad_axis = axis_to_args[args[3]]

        # Compute the gradient
        grad = gradient(args[0], resolution, axis=grad_axis)*args[2]

        return jsum(grad) / count_nonzero(grad)

    @staticmethod
    def compute(
            dose,
            *args):
        """
        Check the jitting status and call the value function.

        Parameters
        ----------
        dose : ndarray
            Dose array.

        *args : tuple
            Optional (non-keyworded) parameters. args[0] should be the dose \
            cube, args[1] the resolution, args[2] the binary segment mask, \
            and args[3] the axis label, e.g. 'x'.

        Returns
        -------
        object of class :class:`~jaxlib.xla_extension.ArrayImpl`
            Dose gradient value.
        """

        # Check if the value function has not yet been jitted
        if not DoseGradient.value_is_jitted:

            # Perform the jitting
            DoseGradient.value_function = jit(
                DoseGradient.value, static_argnums=3)

            # Set 'value_is_jitted' to True
            DoseGradient.value_is_jitted = True

        return DoseGradient.value_function(*args)

    @staticmethod
    def differentiate(
            dose,
            *args):
        """
        Check the jitting status and call the gradient function.

        Parameters
        ----------
        dose : ndarray
            Dose array.

        *args : tuple
            Optional (non-keyworded) parameters. args[0] should be the dose \
            cube, args[1] the resolution, args[2] the binary segment mask, \
            and args[3] the axis label, e.g. 'x'.

        Returns
        -------
        object of class :class:`~jaxlib.xla_extension.ArrayImpl`
            Gradient of the dose gradient.
        """

        # Check if the gradient function has not yet been jitted
        if not DoseGradient.gradient_is_jitted:

            # Perform the jitting
            DoseGradient.gradient_function = jit(grad(
                DoseGradient.value, argnums=0), static_argnums=3)

            # Set 'gradient_is_jitted' to True
            DoseGradient.gradient_is_jitted = True

        return DoseGradient.gradient_function(*args).reshape(-1)
