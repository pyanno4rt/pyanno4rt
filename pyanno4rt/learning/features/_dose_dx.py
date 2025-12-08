"""Dose-volume histogram abscissa feature."""

# Author: Tim Ortkamp

# %% External package import

from jax import grad, jit, lax
from jax.numpy import ceil, floor, int32, sort
from jax.numpy import round as jround

# %% Internal package import

from pyanno4rt.learning.features import DosiomicFeature

# %% Class definition


class DoseDx(DosiomicFeature):
    """Dose-volume histogram abscissa feature class."""

    @staticmethod
    def pyvalue(
            dose,
            level):
        """
        Compute the dose-volume histogram abscissa in 'python' mode.

        Parameters
        ----------
        dose : ndarray
            Dose array.

        level : int
            Reference (relative) volume.

        Returns
        -------
        object of class :class:`~jaxlib.xla_extension.ArrayImpl`
            Dose-volume histogram abscissa value.
        """

        return sort(dose)[int32(jround(len(dose)*(1-level/100)))]

    @staticmethod
    def matvalue(
            dose,
            level):
        """
        Compute the dose-volume histogram abscissa in 'matlab' mode.

        Parameters
        ----------
        dose : ndarray
            Dose array.

        level : int
            Reference (relative) volume.

        Returns
        -------
        object of class :class:`~jaxlib.xla_extension.ArrayImpl`
            Dose-volume histogram abscissa value.
        """

        # Get the quantile
        quantile = len(dose)*(1-level/100)

        # Round the quantile value like MATLAB
        rounded_quantile = lax.cond(
            (quantile-floor(quantile)) != 0.5, jround, ceil, quantile)

        return sort(dose)[int32(rounded_quantile)]

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
            reference (relative) volume.

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
                DoseDx.matvalue if source == 'mat' else DoseDx.pyvalue)

            # Set 'value_is_jitted' to True
            DoseDx.value_is_jitted = True

        return DoseDx.value_function(dose, args[0])

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
            reference (relative) volume.

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
                DoseDx.matvalue if source == 'mat' else DoseDx.pyvalue,
                argnums=0))

            # Set 'gradient_is_jitted' to True
            DoseDx.gradient_is_jitted = True

        return DoseDx.gradient_function(dose, args[0])
