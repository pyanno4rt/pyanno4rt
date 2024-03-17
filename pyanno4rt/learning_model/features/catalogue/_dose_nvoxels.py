"""Dose voxel number feature."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from jax import grad, jit
import jax.numpy as jnp
from scipy.sparse import lil_matrix

# %% Internal package import

from pyanno4rt.learning_model.features.catalogue import DosiomicFeature

# %% Class definition


class DoseNVoxels(DosiomicFeature):
    """Dose voxel number feature class."""

    @staticmethod
    def function(dose):
        """Compute the number of voxels."""
        return jnp.float32(len(dose))

    @staticmethod
    def compute(dose, *args):
        """Check the jitting status and call the computation function."""
        # Check if the value function has not yet been jitted
        if not DoseNVoxels.value_is_jitted:

            # Perform the jitting
            DoseNVoxels.value_function = jit(DoseNVoxels.function)

            # Set 'value_is_jitted' to True for one-time jitting
            DoseNVoxels.value_is_jitted = True

        return DoseNVoxels.value_function(dose)

    @staticmethod
    def differentiate(dose, *args):
        """Check the jitting status and call the differentiation function."""
        # Check if the gradient function has not yet been jitted
        if not DoseNVoxels.gradient_is_jitted:

            # Perform the jitting
            DoseNVoxels.gradient_function = jit(
                grad(DoseNVoxels.function, argnums=0))

            # Set 'gradient_is_jitted' to True for one-time jitting
            DoseNVoxels.gradient_is_jitted = True

        # Initialize the gradient vector
        gradient = lil_matrix((1, args[0]))

        # Insert the gradient values at the indices of the segment
        gradient[:, args[1]] = DoseNVoxels.gradient_function(dose)

        return gradient
