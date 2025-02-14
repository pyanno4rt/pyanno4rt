"""Projections map."""

# Author: Tim Ortkamp

# %% Internal package import

from pyanno4rt.optimization.projections import (
    ConstantRBEProjection, DoseProjection)

# %% Map definition


projection_map = {
    'photon': DoseProjection,
    'proton': ConstantRBEProjection}
