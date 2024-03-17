"""Dose skewness feature."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from jax import grad, jit
import jax.numpy as jnp
from scipy.sparse import lil_matrix

# %% Internal package import

from pyanno4rt.learning_model.features.catalogue import DosiomicFeature

# %% Class definition


class DoseSkewness(DosiomicFeature):
    """Dose skewness feature class."""

    @staticmethod
    def function(dose):
        """Compute the skewness."""
        return 1/len(dose) * jnp.sum(
            jnp.power(dose-jnp.mean(dose), 3)) / jnp.power(jnp.std(dose), 3)

    @staticmethod
    def compute(dose, *args):
        """Check the jitting status and call the computation function."""
        # Check if the value function has not yet been jitted
        if not DoseSkewness.value_is_jitted:

            # Perform the jitting
            DoseSkewness.value_function = jit(DoseSkewness.function)

            # Set 'value_is_jitted' to True for one-time jitting
            DoseSkewness.value_is_jitted = True

        return DoseSkewness.value_function(dose)

    @staticmethod
    def differentiate(dose, *args):
        """Check the jitting status and call the differentiation function."""
        # Check if the gradient function has not yet been jitted
        if not DoseSkewness.gradient_is_jitted:

            # Perform the jitting
            DoseSkewness.gradient_function = jit(
                grad(DoseSkewness.function, argnums=0))

            # Set 'gradient_is_jitted' to True for one-time jitting
            DoseSkewness.gradient_is_jitted = True

        # Initialize the gradient vector
        gradient = lil_matrix((1, args[0]))

        # Insert the gradient values at the indices of the segment
        gradient[:, args[1]] = DoseSkewness.gradient_function(dose)

        return gradient
