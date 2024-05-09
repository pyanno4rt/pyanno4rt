"""Dose gradient feature."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from jax import grad, jit
import jax.numpy as jnp
from scipy.sparse import lil_matrix

# %% Internal package import

from pyanno4rt.learning_model.features.catalogue import DosiomicFeature

# %% Class definition


class DoseGradient(DosiomicFeature):
    """Dose gradient feature class."""

    @staticmethod
    def function(axis, dose, *args):
        """Compute the dose gradient."""
        # Map the axes to their corresponding arguments
        axis_to_args = {'x': [args[1][0], 1],
                        'y': [args[1][1], 0],
                        'z': [args[1][2], 2]}

        # Extract the arguments
        spacing, gradient_axis = axis_to_args[axis]

        # Compute the gradient
        gradient = jnp.gradient(args[0], spacing, axis=gradient_axis) * args[2]

        return jnp.sum(gradient) / jnp.count_nonzero(gradient)

    @staticmethod
    def compute(axis, dose, *args):
        """Check the jitting status and call the computation function."""
        # Check if the value function has not yet been jitted
        if not DoseGradient.value_is_jitted:

            # Perform the jitting
            DoseGradient.value_function = jit(
                DoseGradient.function, static_argnums=0)

            # Set 'value_is_jitted' to True for one-time jitting
            DoseGradient.value_is_jitted = True

        return DoseGradient.value_function(axis, dose, *args)

    @staticmethod
    def differentiate(axis, dose, *args):
        """Check the jitting status and call the differentiation function."""
        # Check if the gradient function has not yet been jitted
        if not DoseGradient.gradient_is_jitted:

            # Perform the jitting
            DoseGradient.gradient_function = jit(
                grad(DoseGradient.function, argnums=2), static_argnums=0)

            # Set 'gradient_is_jitted' to True for one-time jitting
            DoseGradient.gradient_is_jitted = True

        return lil_matrix(DoseGradient.gradient_function(
            axis, dose, *args).reshape(-1))
