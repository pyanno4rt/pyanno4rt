"""Dose skewness feature."""

# Author: Tim Ortkamp

# %% External package import

from jax import grad, jit
import jax.numpy as jnp

# %% Internal package import

from pyanno4rt.learning.features import DosiomicFeature

# %% Class definition


class DoseSkewness(DosiomicFeature):
    """Dose skewness feature class."""

    @staticmethod
    def function(dose):
        """
        Compute the dose skewness.

        Parameters
        ----------
        dose : ndarray
            Dose array.

        Returns
        -------
        object of class :class:`~jaxlib.xla_extension.ArrayImpl`
            Dose skewness value.
        """

        return (
            jnp.sum((dose-jnp.mean(dose))**3) / (jnp.std(dose)**3 * len(dose)))

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
            Dose skewness value.
        """

        # Check if the value function has not yet been jitted
        if not DoseSkewness.value_is_jitted:

            # Perform the jitting
            DoseSkewness.value_function = jit(DoseSkewness.function)

            # Set 'value_is_jitted' to True
            DoseSkewness.value_is_jitted = True

        return DoseSkewness.value_function(dose)

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
            Dose skewness gradient.
        """

        # Check if the gradient function has not yet been jitted
        if not DoseSkewness.gradient_is_jitted:

            # Perform the jitting
            DoseSkewness.gradient_function = jit(grad(
                DoseSkewness.function, argnums=0))

            # Set 'gradient_is_jitted' to True
            DoseSkewness.gradient_is_jitted = True

        return DoseSkewness.gradient_function(dose)
