"""Dose-volume histogram abscissa feature."""

# Author: Tim Ortkamp

# %% External package import

from jax import grad, jit, lax
import jax.numpy as jnp

# %% Internal package import

from pyanno4rt.learning_model.features import DosiomicFeature

# %% Class definition


class DoseDx(DosiomicFeature):
    """Dose-volume histogram abscissa feature class."""

    @staticmethod
    def pyfunction(
            level,
            dose):
        """
        Compute the dose-volume histogram abscissa in 'python' mode.

        Parameters
        ----------
        level : int
            Reference (relative) volume.

        dose : ndarray
            Dose array.

        Returns
        -------
        object of class :class:`~jaxlib.xla_extension.ArrayImpl`
            Dose-volume histogram abscissa value.
        """

        return jnp.sort(dose)[jnp.int32(jnp.round(len(dose)*(1-level/100)))]

    @staticmethod
    def matfunction(
            level,
            dose):
        """
        Compute the dose-volume histogram abscissa in 'matlab' mode.

        Parameters
        ----------
        level : int
            Reference (relative) volume.

        dose : ndarray
            Dose array.

        Returns
        -------
        object of class :class:`~jaxlib.xla_extension.ArrayImpl`
            Dose-volume histogram abscissa value.
        """

        # Get the quantile
        quantile = len(dose)*(1-level/100)

        # Round the quantile value like MATLAB
        rounded_quantile = lax.cond(
            (quantile-jnp.floor(quantile)) != 0.5, jnp.round, jnp.ceil,
            quantile)

        return jnp.sort(dose)[jnp.int32(rounded_quantile)]

    @staticmethod
    def compute(
            level,
            dose,
            *args):
        """
        Check the jitting status and call the value function.

        Parameters
        ----------
        level : int
            Reference (relative) volume.

        dose : ndarray
            Dose array.

        *args : tuple
            Tuple with optional (non-keyworded) parameters.

        Returns
        -------
        object of class :class:`~jaxlib.xla_extension.ArrayImpl`
            Dose-volume histogram ordinate value.
        """

        # Set the calculation mode, either 'mat' or 'py'
        source = 'mat'

        # Check if the value function has not yet been jitted
        if not DoseDx.value_is_jitted:

            # Perform the jitting
            DoseDx.value_function = jit(
                DoseDx.matfunction if source == 'mat' else DoseDx.pyfunction)

            # Set 'value_is_jitted' to True
            DoseDx.value_is_jitted = True

        return DoseDx.value_function(level, dose)

    @staticmethod
    def differentiate(
            level,
            dose,
            *args):
        """
        Check the jitting status and call the gradient function.

        Parameters
        ----------
        level : int
            Reference (relative) volume.

        dose : ndarray
            Dose array.

        *args : tuple
            Tuple with optional (non-keyworded) parameters.

        Returns
        -------
        object of class :class:`~jaxlib.xla_extension.ArrayImpl`
            Dose-volume histogram ordinate value.
        """

        # Set the calculation mode, either 'mat' or 'py'
        source = 'mat'

        # Check if the gradient function has not yet been jitted
        if not DoseDx.gradient_is_jitted:

            # Perform the jitting
            DoseDx.gradient_function = jit(grad(
                DoseDx.matfunction if source == 'mat'
                else DoseDx.pyfunction, argnums=1))

            # Set 'gradient_is_jitted' to True
            DoseDx.gradient_is_jitted = True

        return DoseDx.gradient_function(level, dose)
