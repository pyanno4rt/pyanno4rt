"""Component check map."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from functools import partial

# %% Internal package import

from pyanno4rt.input_check.check_functions import (
    check_length, check_subtype, check_type, check_value, check_value_in_set)

# %% Map definition


component_map = {
    'name': (
        partial(check_type, types=str),
        ),
    'parameter_name': (
        partial(check_type, types=tuple),
        partial(check_subtype, types=str),
        ),
    'parameter_category': (
        partial(check_type, types=tuple),
        partial(check_subtype, types=str),
        ),
    'model_parameters': (
        partial(check_type, types=(type(None), dict)),
        ),
    'embedding': (
        partial(check_type, types=str),
        partial(check_value_in_set, options=('active', 'passive'))
        ),
    'weight': (
        partial(check_type, types=(int, float)),
        partial(check_value, reference=0, sign='>')
        ),
    'rank': (
        partial(check_type, types=int),
        partial(check_value, reference=0, sign='>')),
    'bounds': (
        partial(check_type, types=(type(None), list)),
        partial(check_length, reference=2, sign='=='),
        partial(check_subtype, types=(type(None), int, float))),
    'link': (
        partial(check_type, types=(type(None), list)),
        partial(check_subtype, types=str)
        ),
    'identifier': (
        partial(check_type, types=(type(None), str)),
        ),
    'display': (
        partial(check_type, types=bool),
        ),
    'target_eud': (
        partial(check_type, types=(int, float)),
        partial(check_value, reference=0, sign='>=')
        ),
    'volume_parameter': (
        partial(check_type, types=(int, float)),
        ),
    'target_dose': (
        partial(check_type, types=(int, float)),
        partial(check_value, reference=0, sign='>=')
        ),
    'quantile_volume': (
        partial(check_type, types=(int, float)),
        partial(check_value, reference=0, sign='>='),
        partial(check_value, reference=100, sign='<=')
        ),
    'exponents': (
        partial(check_type, types=list),
        partial(check_subtype, types=(int, float))
        ),
    'maximum_dose': (
        partial(check_type, types=(int, float)),
        partial(check_value, reference=0, sign='>=')
        ),
    'minimum_dose': (
        partial(check_type, types=(int, float)),
        partial(check_value, reference=0, sign='>=')
        ),
    'tolerance_dose_50': (
        partial(check_type, types=(int, float)),
        partial(check_value, reference=0, sign='>=')
        ),
    'slope_parameter': (
        partial(check_type, types=(int, float)),
        ),
    'alpha': (
        partial(check_type, types=(int, float)),
        ),
    'beta': (
        partial(check_type, types=(int, float)),
        )
    }
