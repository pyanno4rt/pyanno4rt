"""Evaluation check map."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from functools import partial

# %% Internal package import

from pyanno4rt.input_check.check_functions import (
    check_subtype, check_type, check_value, check_value_in_set)

# %% Map definition


evaluation_map = {
    'reference_volume': (
        partial(check_type, types=list),
        partial(check_subtype, types=(int, float)),
        partial(check_value, reference=0, sign='>=', is_vector=True),
        partial(check_value, reference=100, sign='<=', is_vector=True)
        ),
    'reference_dose': (
        partial(check_type, types=list),
        partial(check_subtype, types=(int, float)),
        partial(check_value, reference=0, sign='>=', is_vector=True)
        ),
    'dvh_type': (
        partial(check_type, types=str),
        partial(check_value_in_set, options=('cumulative', 'differential'))
        ),
    'number_of_points': (
        partial(check_type, types=int),
        partial(check_value, reference=0, sign='>')
        ),
    'display_metrics': (
        partial(check_type, types=list),
        partial(check_subtype, types=str),
        partial(check_value_in_set, options=(
            'mean', 'std', 'max', 'min', 'Dx', 'Vx', 'CI', 'HI'))
        ),
    'display_segments': (
        partial(check_type, types=list),
        partial(check_subtype, types=str)
        )
    }
