"""Subvolume dose feature."""

# Author: Tim Ortkamp

# %% External package import

from jax import grad
import jax.numpy as jnp

# %% Internal package import

from pyanno4rt.learning_model.features import DosiomicFeature

# %% Class definition


class DoseSubvolume(DosiomicFeature):
    """Subvolume dose feature class."""

    @staticmethod
    def function(
            subvolume,
            _,
            *args):
        """
        Compute the subvolume dose.

        Parameters
        ----------
        subvolume : str
            Subvolume label, e.g. 'x1of2'.

        _ : ndarray
            Dose array.

        *args : tuple
            Tuple with optional (non-keyworded) parameters.

        Returns
        -------
        object of class :class:`~jaxlib.xla_extension.ArrayImpl`
            Subvolume dose value.
        """

        # Map the axes to the permutation orders
        orders = {
            'x': ([1, 0, 2], [1, 2, 0]),
            'y': ([0, 2, 1], [0, 1, 2]),
            'z': ([2, 0, 1], [2, 1, 0])}

        def permute_cube(cube, order):
            """Permute the cube."""

            return jnp.transpose(cube, order[0]), jnp.transpose(cube, order[1])

        # Get the index of the subvolume
        subvolume_index = jnp.int32(subvolume[1])

        # Get the dose cube from the argument
        dose_cube = jnp.array(args[0])

        # Determine the number of values for each subvolume
        number_of_values = jnp.int32(jnp.floor(
            (jnp.count_nonzero(dose_cube)/jnp.int32(subvolume[4]))))

        # Permute the dose cube
        first_permutation, second_permutation = permute_cube(
            dose_cube, orders[subvolume[0]])

        return 1/2 * (
            jnp.mean(
                first_permutation[jnp.nonzero(first_permutation)][
                    ((subvolume_index-1) * number_of_values):(
                        subvolume_index * number_of_values)])
            + jnp.mean(
                second_permutation[jnp.nonzero(second_permutation)][
                    ((subvolume_index-1) * number_of_values):(
                        subvolume_index * number_of_values)]))

    @staticmethod
    def compute(
            subvolume,
            dose,
            *args):
        """
        Check the jitting status and call the value function.

        Parameters
        ----------
        subvolume : str
            Subvolume label, e.g. 'x1of2'.

        dose : ndarray
            Dose array.

        *args : tuple
            Tuple with optional (non-keyworded) parameters.

        Returns
        -------
        object of class :class:`~jaxlib.xla_extension.ArrayImpl`
            Subvolume dose value.
        """

        # Check if the value function has not yet been jitted
        if not DoseSubvolume.value_is_jitted:

            # Perform the jitting
            DoseSubvolume.value_function = DoseSubvolume.function

            # Set 'value_is_jitted' to True
            DoseSubvolume.value_is_jitted = True

        return DoseSubvolume.value_function(subvolume, dose, *args)

    @staticmethod
    def differentiate(
            subvolume,
            dose,
            *args):
        """
        Check the jitting status and call the gradient function.

        Parameters
        ----------
        subvolume : str
            Subvolume label, e.g. 'x1of2'.

        dose : ndarray
            Dose array.

        *args : tuple
            Tuple with optional (non-keyworded) parameters.

        Returns
        -------
        object of class :class:`~jaxlib.xla_extension.ArrayImpl`
            Subvolume dose gradient.
        """

        # Check if the gradient function has not yet been jitted
        if not DoseSubvolume.gradient_is_jitted:

            # Perform the jitting
            DoseSubvolume.gradient_function = grad(
                DoseSubvolume.function, argnums=2)

            # Set 'gradient_is_jitted' to True
            DoseSubvolume.gradient_is_jitted = True

        return DoseSubvolume.gradient_function(
            subvolume, dose, *args).reshape(-1)
