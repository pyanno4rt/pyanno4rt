"""Configuration check map."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from functools import partial

# %% Internal package import

from pyanno4rt.input_check.check_functions import (
    check_extension, check_extensions_in_folder, check_length, check_path,
    check_subtype, check_type, check_value, check_value_in_set)

# %% Map definition


configuration_map = {
    'label': (
        partial(check_type, key_type=str),
        ),
    'min_log_level': (
        partial(check_type, key_type=str),
        partial(check_value_in_set, options=('debug', 'info',
                                             'warning', 'error',
                                             'critical'))
        ),
    'modality': (
        partial(check_type, key_type=str),
        partial(check_value_in_set, options=('photon', 'proton'))
        ),
    'number_of_fractions': (
        partial(check_type, key_type=int),
        partial(check_value, reference=0, sign='>',
                value_group='scalar')
        ),
    'imaging_path': (
        partial(check_type, key_type=str),
        partial(check_path),
        partial(check_extension, extensions=('.mat', '.p')),
        partial(check_extensions_in_folder, extensions=('dcm',))
        ),
    'target_imaging_resolution': (
        partial(check_type, key_type=(type(None), list)),
        partial(check_subtype, key_type=(int, float)),
        partial(check_length, reference=3, sign='=='),
        partial(check_value, reference=0, sign='>',
                value_group='vector')
        ),
    'dose_matrix_path': (
        partial(check_type, key_type=str),
        partial(check_extension, extensions=('.mat', '.npy'))
        ),
    'dose_resolution': (
        partial(check_type, key_type=list),
        partial(check_subtype, key_type=(int, float)),
        partial(check_length, reference=3, sign='=='),
        partial(check_value, reference=0, sign='>',
                value_group='vector')
        )
    }
