"""Dose kurtosis feature."""

# Author: Tim Ortkamp

# %% External package import

from jax import grad, jit
from jax.numpy import mean, std
from jax.numpy import sum as jsum

# %% Internal package import

from pyanno4rt.learning.features import DosiomicFeature

# %% Class definition


class DoseKurtosis(DosiomicFeature):
    """Dose kurtosis feature class."""

    @staticmethod
    def value(dose):
        """
        Compute the dose kurtosis.

        Parameters
        ----------
        dose : ndarray
            Dose array.

        Returns
        -------
        object of class :class:`~jaxlib.xla_extension.ArrayImpl`
            Dose kurtosis value.
        """

        return jsum((dose-mean(dose))**4) / (std(dose)**4 * len(dose))

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
            Optional (non-keyworded) parameters.

        Returns
        -------
        object of class :class:`~jaxlib.xla_extension.ArrayImpl`
            Dose kurtosis value.
        """

        # Check if the value function has not yet been jitted
        if not DoseKurtosis.value_is_jitted:

            # Perform the jitting
            DoseKurtosis.value_function = jit(DoseKurtosis.value)

            # Set 'value_is_jitted' to True
            DoseKurtosis.value_is_jitted = True

        return DoseKurtosis.value_function(dose)

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
            Optional (non-keyworded) parameters.

        Returns
        -------
        object of class :class:`~jaxlib.xla_extension.ArrayImpl`
            Dose kurtosis gradient.
        """

        # Check if the gradient function has not yet been jitted
        if not DoseKurtosis.gradient_is_jitted:

            # Perform the jitting
            DoseKurtosis.gradient_function = jit(grad(
                DoseKurtosis.value, argnums=0))

            # Set 'gradient_is_jitted' to True
            DoseKurtosis.gradient_is_jitted = True

        return DoseKurtosis.gradient_function(dose)
