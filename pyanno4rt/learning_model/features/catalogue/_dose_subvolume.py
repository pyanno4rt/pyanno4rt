"""Subvolume dose feature."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from jax import grad
import jax.numpy as jnp
from scipy.sparse import lil_matrix

# %% Internal package import

from pyanno4rt.learning_model.features.catalogue import DosiomicFeature

# %% Class definition


class DoseSubvolume(DosiomicFeature):
    """Subvolume dose feature class."""

    @staticmethod
    def function(subvolume, _, *args):
        """Compute the subvolume dose."""
        # Map the axes to the permutation orders
        orders = {'x': ([1, 0, 2], [1, 2, 0]),
                  'y': ([0, 2, 1], [0, 1, 2]),
                  'z': ([2, 0, 1], [2, 1, 0])}

        def transpose_cube(cube, order):
            """Transpose the cube in a specific order."""
            # Get the first permutation
            first_permutation = jnp.transpose(cube, order[0])

            # Get the second permutation
            second_permutation = jnp.transpose(cube, order[1])

            return first_permutation, second_permutation

        # Get the axis from the argument
        axis = subvolume[0]

        # Get the index of the subvolume as an integer
        subvolume_index = jnp.int32(subvolume[1])

        # Get the dose cube from the argument
        dose_cube = jnp.array(args[0])

        # Determine the number of values for each subvolume
        number_of_values = jnp.int32(jnp.floor(
            (jnp.count_nonzero(dose_cube)/jnp.int32(subvolume[4]))))

        # Transpose the dose cube to yield the permutations
        first_permutation, second_permutation = transpose_cube(
            dose_cube, orders[axis])

        return 1/2 * (
            jnp.mean(first_permutation[jnp.nonzero(first_permutation)][
                ((subvolume_index-1) * number_of_values):(
                    subvolume_index * number_of_values)])
            + jnp.mean(second_permutation[jnp.nonzero(second_permutation)][
                ((subvolume_index-1) * number_of_values):(
                    subvolume_index * number_of_values)]))

    @staticmethod
    def compute(subvolume, dose, *args):
        """Check the jitting status and call the computation function."""
        # Check if the value function has not yet been jitted
        if not DoseSubvolume.value_is_jitted:

            # Perform the jitting
            DoseSubvolume.value_function = DoseSubvolume.function

            # Set 'value_is_jitted' to True for one-time jitting
            DoseSubvolume.value_is_jitted = True

        return DoseSubvolume.value_function(subvolume, dose, *args)

    @staticmethod
    def differentiate(subvolume, dose, *args):
        """Check the jitting status and call the differentiation function."""
        # Check if the gradient function has not yet been jitted
        if not DoseSubvolume.gradient_is_jitted:

            # Perform the jitting
            DoseSubvolume.gradient_function = grad(
                DoseSubvolume.function, argnums=2)

            # Set 'gradient_is_jitted' to True for one-time jitting
            DoseSubvolume.gradient_is_jitted = True

        return lil_matrix(DoseSubvolume.gradient_function(
            subvolume, dose, *args).reshape(-1))
