"""Dose mean feature."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from jax import grad, jit
import jax.numpy as jnp
from scipy.sparse import lil_matrix

# %% Internal package import

from pyanno4rt.learning_model.features.catalogue import DosiomicFeature

# %% Class definition


class DoseMean(DosiomicFeature):
    """Dose mean feature class."""

    @staticmethod
    def function(dose):
        """Compute the mean dose."""
        return jnp.mean(dose)

    @staticmethod
    def compute(dose, *args):
        """Check the jitting status and call the computation function."""
        # Check if the value function has not yet been jitted
        if not DoseMean.value_is_jitted:

            # Perform the jitting
            DoseMean.value_function = jit(DoseMean.function)

            # Set 'value_is_jitted' to True for one-time jitting
            DoseMean.value_is_jitted = True

        return DoseMean.value_function(dose)

    @staticmethod
    def differentiate(dose, *args):
        """Check the jitting status and call the differentiation function."""
        # Check if the gradient function has not yet been jitted
        if not DoseMean.gradient_is_jitted:

            # Perform the jitting
            DoseMean.gradient_function = jit(
                grad(DoseMean.function, argnums=0))

            # Set 'gradient_is_jitted' to True for one-time jitting
            DoseMean.gradient_is_jitted = True

        # Initialize the gradient vector
        gradient = lil_matrix((1, args[0]))

        # Insert the gradient values at the indices of the segment
        gradient[:, args[1]] = DoseMean.gradient_function(dose)

        return gradient
