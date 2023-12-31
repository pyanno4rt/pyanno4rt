"""Segment area feature."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

import jax.numpy as jnp

# %% Internal package import

from pyanno4rt.learning_model.features.catalogue import RadiomicFeature

# %% Class definition


class SegmentArea(RadiomicFeature):
    """Segment area feature class."""

    @staticmethod
    def compute(mask, spacing):
        """Compute the area."""

        def compute_directional_area(parameters):
            """Compute the area in one direction."""
            # Compute the function terms
            terms = jnp.array([jnp.sum(jnp.sum(parameter[2]))
                               if parameter[0] in (0, parameters[0]-1)
                               else jnp.sum(jnp.sum(
                                   abs(parameter[1]-parameter[2])))
                               for parameter in parameters[1]])

            return jnp.sum(terms) * parameters[2]

        # Set the parameters for the area computation in all three axes
        parameters = ((mask.shape[0], ((i, mask[i+1, :, :], mask[i, :, :])
                                       for i in range(mask.shape[0]-1)),
                       spacing[1] * spacing[2]),
                      (mask.shape[1], ((j, mask[:, j+1, :], mask[:, j, :])
                                       for j in range(mask.shape[1]-1)),
                       spacing[0] * spacing[2]),
                      (mask.shape[2], ((k, mask[:, :, k+1], mask[:, :, k])
                                       for k in range(mask.shape[2]-1)),
                       spacing[0] * spacing[1]))

        return jnp.sum(jnp.array([compute_directional_area(parameter)
                                  for parameter in parameters]))
