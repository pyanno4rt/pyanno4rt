"""Projections map."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% Internal package import

from pyanno4rt.optimization.projections import (
    ConstantRBEProjection, DoseProjection)

# %% Map definition


projections_map = {'photon': DoseProjection,
                   'proton': ConstantRBEProjection}
