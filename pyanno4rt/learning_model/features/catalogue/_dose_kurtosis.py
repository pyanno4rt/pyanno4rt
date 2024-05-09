"""Dose kurtosis feature."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from jax import grad, jit
import jax.numpy as jnp
from scipy.sparse import lil_matrix

# %% Internal package import

from pyanno4rt.learning_model.features.catalogue import DosiomicFeature

# %% Class definition


class DoseKurtosis(DosiomicFeature):
    """Dose kurtosis feature class."""

    @staticmethod
    def function(dose):
        """Compute the kurtosis."""
        return 1/len(dose) * jnp.sum(
            jnp.power(dose-jnp.mean(dose), 4)) / jnp.power(jnp.std(dose), 4)

    @staticmethod
    def compute(dose, *args):
        """Check the jitting status and call the computation function."""
        # Check if the value function has not yet been jitted
        if not DoseKurtosis.value_is_jitted:

            # Perform the jitting
            DoseKurtosis.value_function = jit(DoseKurtosis.function)

            # Set 'value_is_jitted' to True for one-time jitting
            DoseKurtosis.value_is_jitted = True

        return DoseKurtosis.value_function(dose)

    @staticmethod
    def differentiate(dose, *args):
        """Check the jitting status and call the differentiation function."""
        # Check if the value function has not yet been jitted
        if not DoseKurtosis.gradient_is_jitted:

            # Perform the jitting
            DoseKurtosis.gradient_function = jit(
                grad(DoseKurtosis.function, argnums=0))

            # Set 'gradient_is_jitted' to True for one-time jitting
            DoseKurtosis.gradient_is_jitted = True

        # Initialize the gradient vector
        gradient = lil_matrix((1, args[0]))

        # Insert the gradient values at the indices of the segment
        gradient[:, args[1]] = DoseKurtosis.gradient_function(dose)

        return gradient
