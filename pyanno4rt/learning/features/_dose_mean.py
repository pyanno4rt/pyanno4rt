"""Dose mean feature."""

# Author: Tim Ortkamp

# %% External package import

from jax import grad, jit
from jax.numpy import mean

# %% Internal package import

from pyanno4rt.learning.features import DosiomicFeature

# %% Class definition


class DoseMean(DosiomicFeature):
    """Dose mean feature class."""

    @staticmethod
    def value(dose):
        """
        Compute the mean dose.

        Parameters
        ----------
        dose : ndarray
            Dose array.

        Returns
        -------
        object of class :class:`~jaxlib.xla_extension.ArrayImpl`
            Mean dose value.
        """

        return mean(dose)

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
            Mean dose value.
        """

        # Check if the value function has not yet been jitted
        if not DoseMean.value_is_jitted:

            # Perform the jitting
            DoseMean.value_function = jit(DoseMean.value)

            # Set 'value_is_jitted' to True
            DoseMean.value_is_jitted = True

        return DoseMean.value_function(dose)

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
            Mean dose gradient.
        """

        # Check if the gradient function has not yet been jitted
        if not DoseMean.gradient_is_jitted:

            # Perform the jitting
            DoseMean.gradient_function = jit(grad(DoseMean.value, argnums=0))

            # Set 'gradient_is_jitted' to True
            DoseMean.gradient_is_jitted = True

        return DoseMean.gradient_function(dose)
