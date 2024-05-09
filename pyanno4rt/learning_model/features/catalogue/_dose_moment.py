"""Dose moment feature."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from jax import grad, jit
import jax.numpy as jnp
from scipy.sparse import lil_matrix

# %% Internal package import

from pyanno4rt.learning_model.features.catalogue import DosiomicFeature

# %% Class definition


class DoseMoment(DosiomicFeature):
    """Dose moment feature class."""

    @staticmethod
    def function(coefficients, _, *args):
        """Compute the dose moment."""

        def compute_scaled_cube(array):
            """Compute the mean of the dose cube scaled by an array."""
            return jnp.sum((array * dose_cube)) / jnp.sum(dose_cube)

        # Get the dose cube from the argument
        dose_cube = args[0]

        # Get the coefficients of the moment function from the argument
        coeff_1, coeff_2, coeff_3 = tuple(map(jnp.int32, coefficients))

        # Determine the axis points from a meshed grid
        points_x, points_y, points_z = jnp.meshgrid(
            jnp.array(range(dose_cube.shape[0])),
            jnp.array(range(dose_cube.shape[1])),
            jnp.array(range(dose_cube.shape[2])))

        # Compute the means of the axis points
        mean_x, mean_y, mean_z = tuple(
            map(compute_scaled_cube, (points_x, points_y, points_z)))

        # Compute the moment function numerator
        numerator = jnp.sum((points_x-mean_x)**coeff_1
                            * (points_y-mean_y)**coeff_2
                            * (points_z-mean_z)**coeff_3
                            * dose_cube)

        # Compute the moment function denominator
        denominator = jnp.sum(dose_cube)**((coeff_1+coeff_2+coeff_3)/3 + 1)

        return numerator/denominator

    @staticmethod
    def compute(coefficients, dose, *args):
        """Check the jitting status and call the computation function."""
        # Check if the value function has not yet been jitted
        if not DoseMoment.value_is_jitted:

            # Perform the jitting
            DoseMoment.value_function = jit(
                DoseMoment.function, static_argnums=0)

            # Set 'value_is_jitted' to True for one-time jitting
            DoseMoment.value_is_jitted = True

        return DoseMoment.value_function(coefficients, dose, *args)

    @staticmethod
    def differentiate(coefficients, dose, *args):
        """Check the jitting status and call the differentiation function."""
        # Check if the gradient function has not yet been jitted
        if not DoseMoment.gradient_is_jitted:

            # Perform the jitting
            DoseMoment.gradient_function = jit(
                grad(DoseMoment.function, argnums=2), static_argnums=0)

            # Set 'gradient_is_jitted' to True for one-time jitting
            DoseMoment.gradient_is_jitted = True

        return lil_matrix(DoseMoment.gradient_function(
            coefficients, dose, *args).reshape(-1))
