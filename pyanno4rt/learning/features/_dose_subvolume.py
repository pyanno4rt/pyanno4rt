"""Subvolume dose feature."""

# Author: Tim Ortkamp

# %% External package import

from jax import grad
from jax.numpy import (
    array, count_nonzero, floor, int32, mean, nonzero, transpose)

# %% Internal package import

from pyanno4rt.learning.features import DosiomicFeature

# %% Class definition


class DoseSubvolume(DosiomicFeature):
    """Subvolume dose feature class."""

    @staticmethod
    def value(
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
            Optional (non-keyworded) parameters. args[0] should be the dose \
            cube, args[1] the resolution, and args[2] the binary segment mask.

        Returns
        -------
        object of class :class:`~jaxlib.xla_extension.ArrayImpl`
            Subvolume dose value.
        """

        def permute_cube(cube, order):
            """Permute the cube."""

            return transpose(cube, order[0]), transpose(cube, order[1])

        # Map the axes to the permutation orders
        orders = {
            'x': ([1, 0, 2], [1, 2, 0]),
            'y': ([0, 2, 1], [0, 1, 2]),
            'z': ([2, 0, 1], [2, 1, 0])}

        # Get the dose cube
        dose_cube = array(args[0])

        # Get the subvolume index
        subvolume_index = int32(subvolume[1])

        # Determine the number of values for each subvolume
        number_of_values = int32(floor(
            (count_nonzero(dose_cube)/int32(subvolume[4]))))

        # Permute the dose cube
        first_permutation, second_permutation = permute_cube(
            dose_cube, orders[subvolume[0]])

        return 0.5 * (
            mean(
                first_permutation[nonzero(first_permutation)][
                    ((subvolume_index-1) * number_of_values):(
                        subvolume_index * number_of_values)])
            + mean(
                second_permutation[nonzero(second_permutation)][
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

        _ : ndarray
            Dose array.

        *args : tuple
            Optional (non-keyworded) parameters. args[0] should be the dose \
            cube, args[1] the resolution, and args[2] the binary segment mask.

        Returns
        -------
        object of class :class:`~jaxlib.xla_extension.ArrayImpl`
            Subvolume dose value.
        """

        # Check if the value function has not yet been jitted
        if not DoseSubvolume.value_is_jitted:

            # Perform the jitting
            DoseSubvolume.value_function = DoseSubvolume.value

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

        _ : ndarray
            Dose array.

        *args : tuple
            Optional (non-keyworded) parameters. args[0] should be the dose \
            cube, args[1] the resolution, and args[2] the binary segment mask.

        Returns
        -------
        object of class :class:`~jaxlib.xla_extension.ArrayImpl`
            Subvolume dose gradient.
        """

        # Check if the gradient function has not yet been jitted
        if not DoseSubvolume.gradient_is_jitted:

            # Perform the jitting
            DoseSubvolume.gradient_function = grad(
                DoseSubvolume.value, argnums=2)

            # Set 'gradient_is_jitted' to True
            DoseSubvolume.gradient_is_jitted = True

        return DoseSubvolume.gradient_function(
            subvolume, dose, *args).reshape(-1)
