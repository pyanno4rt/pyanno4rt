"""Dose maximum feature."""

# Author: Tim Ortkamp

# %% External package import

from jax import grad, jit
import jax.numpy as jnp

# %% Internal package import

from pyanno4rt.learning.features import DosiomicFeature

# %% Class definition


class DoseMaximum(DosiomicFeature):
    """Dose maximum feature class."""

    @staticmethod
    def function(dose):
        """
        Compute the maximum dose.

        Parameters
        ----------
        dose : ndarray
            Dose array.

        Returns
        -------
        object of class :class:`~jaxlib.xla_extension.ArrayImpl`
            Maximum dose value.
        """

        return jnp.max(dose)

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
            Maximum dose value.
        """

        # Check if the value function has not yet been jitted
        if not DoseMaximum.value_is_jitted:

            # Perform the jitting
            DoseMaximum.value_function = jit(DoseMaximum.function)

            # Set 'value_is_jitted' to True
            DoseMaximum.value_is_jitted = True

        return DoseMaximum.value_function(dose)

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
            Maximum dose gradient.
        """

        # Check if the gradient function has not yet been jitted
        if not DoseMaximum.gradient_is_jitted:

            # Perform the jitting
            DoseMaximum.gradient_function = jit(grad(
                DoseMaximum.function, argnums=0))

            # Set 'gradient_is_jitted' to True
            DoseMaximum.gradient_is_jitted = True

        return DoseMaximum.gradient_function(dose)
