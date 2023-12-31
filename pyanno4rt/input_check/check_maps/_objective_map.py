"""Objective check map."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from functools import partial

# %% Internal package import

from pyanno4rt.input_check.check_functions import (
    check_subtype, check_type, check_value, check_value_in_set)

# %% Map definition


objective_map = {
    'name': (
        partial(check_type, key_type=str),
        ),
    'parameter_name': (
        partial(check_type, key_type=tuple),
        partial(check_subtype, key_type=str),
        ),
    'parameter_category': (
        partial(check_type, key_type=tuple),
        partial(check_subtype, key_type=str),
        ),
    'model_parameters': (
        partial(check_type, key_type=(type(None), dict)),
        ),
    'embedding': (
        partial(check_type, key_type=str),
        partial(check_value_in_set, options=('active', 'passive'))
        ),
    'weight': (
        partial(check_type, key_type=(int, float)),
        partial(check_value, reference=0, sign='>',
                value_group='scalar')
        ),
    'link': (
        partial(check_type, key_type=(type(None), list)),
        partial(check_subtype, key_type=str, type_group=list)
        ),
    'identifier': (
        partial(check_type, key_type=(type(None), str)),
        ),
    'display': (
        partial(check_type, key_type=bool),
        ),
    'target_eud': (
        partial(check_type, key_type=(int, float)),
        partial(check_value, reference=0, sign='>=',
                value_group='scalar')
        ),
    'volume_parameter': (
        partial(check_type, key_type=(int, float))
        ),
    'target_dose': (
        partial(check_type, key_type=(int, float)),
        partial(check_value, reference=0, sign='>=',
                value_group='scalar')
        ),
    'maximum_volume': (
        partial(check_type, key_type=(int, float)),
        partial(check_value, reference=0, sign='>=',
                value_group='scalar'),
        partial(check_value, reference=100, sign='<=',
                value_group='scalar')
        ),
    'minimum_volume': (
        partial(check_type, key_type=(int, float)),
        partial(check_value, reference=0, sign='>=',
                value_group='scalar'),
        partial(check_value, reference=100, sign='<=',
                value_group='scalar')
        ),
    'exponents': (
        partial(check_type, key_type=list),
        partial(check_subtype, key_type=float)
        ),
    'maximum_dose': (
        partial(check_type, key_type=(int, float)),
        partial(check_value, reference=0, sign='>=',
                value_group='scalar')
        ),
    'minimum_dose': (
        partial(check_type, key_type=(int, float)),
        partial(check_value, reference=0, sign='>=',
                value_group='scalar')
        )
    }
