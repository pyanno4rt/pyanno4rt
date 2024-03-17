"""Dose deviation feature."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from jax import grad, jit
import jax.numpy as jnp
from scipy.sparse import lil_matrix

# %% Internal package import

from pyanno4rt.learning_model.features.catalogue import DosiomicFeature

# %% Class definition


class DoseDeviation(DosiomicFeature):
    """Dose deviation feature class."""

    @staticmethod
    def function(dose):
        """Compute the standard deviation of the dose."""
        return jnp.std(dose)

    @staticmethod
    def compute(dose, *args):
        """Check the jitting status and call the computation function."""
        # Check if the value function has not yet been jitted
        if not DoseDeviation.value_is_jitted:

            # Perform the jitting
            DoseDeviation.value_function = jit(DoseDeviation.function)

            # Set 'value_is_jitted' to True for one-time jitting
            DoseDeviation.value_is_jitted = True

        return DoseDeviation.value_function(dose)

    @staticmethod
    def differentiate(dose, *args):
        """Check the jitting status and call the differentiation function."""
        # Check if the gradient function has not yet been jitted
        if not DoseDeviation.gradient_is_jitted:

            # Perform the jitting
            DoseDeviation.gradient_function = jit(
                grad(DoseDeviation.function, argnums=0))

            # Set 'gradient_is_jitted' to True for one-time jitting
            DoseDeviation.gradient_is_jitted = True

        # Initialize the gradient vector
        gradient = lil_matrix((1, args[0]))

        # Insert the gradient values at the indices of the segment
        gradient[:, args[1]] = DoseDeviation.gradient_function(dose)

        return gradient
