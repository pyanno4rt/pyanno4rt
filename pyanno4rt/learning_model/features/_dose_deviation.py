"""Dose deviation feature."""

# Author: Tim Ortkamp

# %% External package import

from jax import grad, jit
import jax.numpy as jnp

# %% Internal package import

from pyanno4rt.learning_model.features import DosiomicFeature

# %% Class definition


class DoseDeviation(DosiomicFeature):
    """Dose deviation feature class."""

    @staticmethod
    def function(dose):
        """
        Compute the dose deviation.

        Parameters
        ----------
        dose : ndarray
            Dose array.

        Returns
        -------
        object of class :class:`~jaxlib.xla_extension.ArrayImpl`
            Dose deviation value.
        """

        return jnp.std(dose)

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
            Dose deviation value.
        """

        # Check if the value function has not yet been jitted
        if not DoseDeviation.value_is_jitted:

            # Perform the jitting
            DoseDeviation.value_function = jit(DoseDeviation.function)

            # Set 'value_is_jitted' to True
            DoseDeviation.value_is_jitted = True

        return DoseDeviation.value_function(dose)

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
            Dose deviation gradient.
        """

        # Check if the gradient function has not yet been jitted
        if not DoseDeviation.gradient_is_jitted:

            # Perform the jitting
            DoseDeviation.gradient_function = jit(grad(
                DoseDeviation.function, argnums=0))

            # Set 'gradient_is_jitted' to True
            DoseDeviation.gradient_is_jitted = True

        return DoseDeviation.gradient_function(dose)
