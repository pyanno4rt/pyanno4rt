"""Dose kurtosis feature."""

# Author: Tim Ortkamp

# %% External package import

from jax import grad, jit
import jax.numpy as jnp

# %% Internal package import

from pyanno4rt.learning.features import DosiomicFeature

# %% Class definition


class DoseKurtosis(DosiomicFeature):
    """Dose kurtosis feature class."""

    @staticmethod
    def function(dose):
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

        return (
            jnp.sum((dose-jnp.mean(dose))**4) / (jnp.std(dose)**4 * len(dose)))

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
            Tuple with optional (non-keyworded) parameters.

        Returns
        -------
        object of class :class:`~jaxlib.xla_extension.ArrayImpl`
            Dose kurtosis value.
        """

        # Check if the value function has not yet been jitted
        if not DoseKurtosis.value_is_jitted:

            # Perform the jitting
            DoseKurtosis.value_function = jit(DoseKurtosis.function)

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
            Tuple with optional (non-keyworded) parameters.

        Returns
        -------
        object of class :class:`~jaxlib.xla_extension.ArrayImpl`
            Dose kurtosis gradient.
        """

        # Check if the gradient function has not yet been jitted
        if not DoseKurtosis.gradient_is_jitted:

            # Perform the jitting
            DoseKurtosis.gradient_function = jit(grad(
                DoseKurtosis.function, argnums=0))

            # Set 'gradient_is_jitted' to True
            DoseKurtosis.gradient_is_jitted = True

        return DoseKurtosis.gradient_function(dose)
