"""Dose voxel number feature."""

# Author: Tim Ortkamp

# %% External package import

from jax import grad, jit
import jax.numpy as jnp

# %% Internal package import

from pyanno4rt.learning_model.features import DosiomicFeature

# %% Class definition


class DoseNVoxels(DosiomicFeature):
    """Dose voxel number feature class."""

    @staticmethod
    def function(dose):
        """
        Compute the number of dose voxels.

        Parameters
        ----------
        dose : ndarray
            Dose array.

        Returns
        -------
        object of class :class:`~jaxlib.xla_extension.ArrayImpl`
            Dose voxel number.
        """

        return jnp.float32(len(dose))

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
            Dose voxel number.
        """

        # Check if the value function has not yet been jitted
        if not DoseNVoxels.value_is_jitted:

            # Perform the jitting
            DoseNVoxels.value_function = jit(DoseNVoxels.function)

            # Set 'value_is_jitted' to True
            DoseNVoxels.value_is_jitted = True

        return DoseNVoxels.value_function(dose)

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
            Dose voxel number gradient.
        """

        # Check if the gradient function has not yet been jitted
        if not DoseNVoxels.gradient_is_jitted:

            # Perform the jitting
            DoseNVoxels.gradient_function = jit(grad(
                DoseNVoxels.function, argnums=0))

            # Set 'gradient_is_jitted' to True
            DoseNVoxels.gradient_is_jitted = True

        return DoseNVoxels.gradient_function(dose)
