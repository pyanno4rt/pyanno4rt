"""Configuration check map."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from functools import partial

# %% Internal package import

from pyanno4rt.input_check.check_functions import (
    check_length, check_regular_extension, check_regular_extension_directory,
    check_subtype, check_type, check_value, check_value_in_set)

# %% Map definition


configuration_map = {
    'label': (
        partial(check_type, types=str),
        ),
    'min_log_level': (
        partial(check_type, types=str),
        partial(check_value_in_set, options=(
            'debug', 'info', 'warning', 'error', 'critical'))
        ),
    'modality': (
        partial(check_type, types=str),
        partial(check_value_in_set, options=('photon', 'proton'))
        ),
    'number_of_fractions': (
        partial(check_type, types=int),
        partial(check_value, reference=0, sign='>')
        ),
    'imaging_path': (
        partial(check_type, types=str),
        partial(check_regular_extension, extensions=('.mat', '.p')),
        partial(check_regular_extension_directory, extensions=('dcm',))
        ),
    'target_imaging_resolution': (
        partial(check_type, types=(type(None), list)),
        partial(check_subtype, types=(int, float)),
        partial(check_length, reference=3, sign='=='),
        partial(check_value, reference=0, sign='>', is_vector=True)
        ),
    'dose_matrix_path': (
        partial(check_type, types=str),
        partial(check_regular_extension, extensions=('.mat', '.npy'))
        ),
    'dose_resolution': (
        partial(check_type, types=list),
        partial(check_subtype, types=(int, float)),
        partial(check_length, reference=3, sign='=='),
        partial(check_value, reference=0, sign='>', is_vector=True)
        )
    }
