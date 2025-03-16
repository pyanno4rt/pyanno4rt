"""Dose minimum feature."""

# Author: Tim Ortkamp

# %% External package import

from jax import grad, jit
import jax.numpy as jnp

# %% Internal package import

from pyanno4rt.learning_model.features import DosiomicFeature

# %% Class definition


class DoseMinimum(DosiomicFeature):
    """Dose minimum feature class."""

    @staticmethod
    def function(dose):
        """
        Compute the minimum dose.

        Parameters
        ----------
        dose : ndarray
            Dose array.

        Returns
        -------
        object of class :class:`~jaxlib.xla_extension.ArrayImpl`
            Minimum dose value.
        """

        return jnp.min(dose)

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
            Minimum dose value.
        """

        # Check if the value function has not yet been jitted
        if not DoseMinimum.value_is_jitted:

            # Perform the jitting
            DoseMinimum.value_function = jit(DoseMinimum.function)

            # Set 'value_is_jitted' to True
            DoseMinimum.value_is_jitted = True

        return DoseMinimum.value_function(dose)

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
            Minimum dose gradient.
        """

        # Check if the gradient function has not yet been jitted
        if not DoseMinimum.gradient_is_jitted:

            # Perform the jitting
            DoseMinimum.gradient_function = jit(grad(
                DoseMinimum.function, argnums=0))

            # Set 'gradient_is_jitted' to True
            DoseMinimum.gradient_is_jitted = True

        return DoseMinimum.gradient_function(dose)
