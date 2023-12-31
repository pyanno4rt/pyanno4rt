"""Top-level check map."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from functools import partial

# %% Internal package import

from pyanno4rt.input_check.check_functions import check_type

# %% Map definition


top_level_map = {
    'configuration': (
        partial(check_type, key_type=dict),
        ),
    'optimization': (
        partial(check_type, key_type=dict),
        ),
    'evaluation': (
        partial(check_type, key_type=dict),
        )
    }
