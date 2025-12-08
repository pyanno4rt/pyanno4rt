"""Dose-volume histogram ordinate feature."""

# Author: Tim Ortkamp

# %% External package import

from jax import grad, jit
from jax.numpy import sum as jsum

# %% Internal package import

from pyanno4rt.learning.features import DosiomicFeature

# %% Class definition


class DoseVx(DosiomicFeature):
    """Dose-volume histogram ordinate feature class."""

    @staticmethod
    def value(
            dose,
            level):
        """
        Compute the dose-volume histogram ordinate.

        Parameters
        ----------
        dose : ndarray
            Dose array.

        level : int or float
            Reference dose level.

        Returns
        -------
        object of class :class:`~jaxlib.xla_extension.ArrayImpl`
            Dose-volume histogram ordinate value.
        """

        return jsum(dose >= level) / len(dose)

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
            Optional (non-keyworded) parameters. args[0] should be the \
            reference dose level.

        Returns
        -------
        object of class :class:`~jaxlib.xla_extension.ArrayImpl`
            Dose-volume histogram ordinate value.
        """

        # Check if the value function has not yet been jitted
        if not DoseVx.value_is_jitted:

            # Perform the jitting
            DoseVx.value_function = jit(DoseVx.value)

            # Set 'value_is_jitted' to True
            DoseVx.value_is_jitted = True

        return DoseVx.value_function(dose, args[0])

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
            Optional (non-keyworded) parameters. args[0] should be the \
            reference dose level.

        Returns
        -------
        object of class :class:`~jaxlib.xla_extension.ArrayImpl`
            Dose-volume histogram ordinate gradient.
        """

        # Check if the gradient function has not yet been jitted
        if not DoseVx.gradient_is_jitted:

            # Perform the jitting
            DoseVx.gradient_function = jit(grad(DoseVx.value, argnums=0))

            # Set 'gradient_is_jitted' to True
            DoseVx.gradient_is_jitted = True

        return DoseVx.gradient_function(dose, args[0])
