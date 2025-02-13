"""Dose-volume histogram ordinate feature."""

# Author: Tim Ortkamp

# %% External package import

from jax import grad, jit
import jax.numpy as jnp

# %% Internal package import

from pyanno4rt.learning_model.features import DosiomicFeature

# %% Class definition


class DoseVx(DosiomicFeature):
    """Dose-volume histogram ordinate feature class."""

    @staticmethod
    def function(level, dose):
        """Compute the dose-volume histogram ordinate."""

        return jnp.sum(dose >= level) / len(dose)

    @staticmethod
    def compute(level, dose, *args):
        """Check the jitting status and call the computation function."""

        # Check if the value function has not yet been jitted
        if not DoseVx.value_is_jitted:

            # Perform the jitting
            DoseVx.value_function = jit(DoseVx.function)

            # Set 'gradient_is_jitted' to True for one-time jitting
            DoseVx.value_is_jitted = True

        return DoseVx.value_function(level, dose)

    @staticmethod
    def differentiate(level, dose, *args):
        """Check the jitting status and call the differentiation function."""

        # Check if the gradient function has not yet been jitted
        if not DoseVx.gradient_is_jitted:

            # Perform the jitting
            DoseVx.gradient_function = jit(grad(DoseVx.function, argnums=1))

            # Set 'gradient_is_jitted' to True for one-time jitting
            DoseVx.gradient_is_jitted = True

        return DoseVx.gradient_function(level, dose)
