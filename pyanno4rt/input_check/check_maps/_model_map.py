"""Model parameter check map."""

# Author: Tim Ortkamp

# %% External package import

from functools import partial

# %% Internal package import

from pyanno4rt.input_check.check_functions import (
    check_data_columns, check_key_in_dict, check_length, check_path,
    check_regular_extension, check_regular_extension_directory, check_subtype,
    check_type, check_string_is_number, check_value, check_value_in_set)
from pyanno4rt.learning_model.features import feature_map
from pyanno4rt.learning_model.losses import loss_map
from pyanno4rt.learning_model.preprocessing.cleaners import cleaner_map
from pyanno4rt.learning_model.preprocessing.reducers import reducer_map
from pyanno4rt.learning_model.preprocessing.samplers import sampler_map
from pyanno4rt.learning_model.preprocessing.transformers import transformer_map

# %% Map definition


model_map = {
    'model_label': (
        partial(check_type, types=str),
        ),
    'model_folder_path': (
        partial(check_type, types=(type(None), str)),
        partial(check_path)
        ),
    'data_path': (
        partial(check_type, types={True: (type(None), str), False: str}),
        partial(check_regular_extension, extensions=('.csv',)),
        partial(check_regular_extension_directory, extensions=(
            'jpg', 'npy', 'npz', 'png'), no_directory=('.csv',))
        ),
    'data_columns': (
        partial(check_type, types={True: (type(None), dict), False: dict}),
        partial(check_data_columns, check_functions=(
            check_value_in_set,
            partial(check_type, types=dict),
            partial(check_key_in_dict, keys=('type',)),
            partial(check_type, types=str),
            partial(check_value_in_set, options=('feature', 'label')),
            partial(check_key_in_dict, keys=(
                'scale', 'value', 'function', 'segment')),
            partial(check_type, types=str),
            partial(check_value_in_set, options=(
                'metric', 'nominal', 'ordinal')),
            partial(check_type, types=(type(None), int, float, str)),
            partial(check_type, types=str),
            partial(check_value_in_set, options=tuple(feature_map)),
            partial(check_key_in_dict, keys=('argument',)),
            partial(check_type, types=(int, float)),
            partial(check_value, reference=1, sign='>='),
            partial(check_value, reference=99, sign='<='),
            partial(check_type, types=(int, float)),
            partial(check_value, reference=0, sign='>'),
            partial(check_value, reference=100, sign='<'),
            partial(check_type, types=str),
            partial(check_value_in_set, options=('x', 'y', 'z')),
            partial(check_type, types=str),
            partial(check_length, reference=3, sign='=='),
            check_string_is_number,
            partial(check_value, reference=0, sign='>'),
            partial(check_type, types=str),
            partial(check_value_in_set, options=(
                'x1of2', 'x2of2', 'x1of3', 'x2of3', 'x3of3', 'y1of2', 'y2of2',
                'y1of3', 'y2of3', 'y3of3', 'z1of2', 'z2of2', 'z1of3', 'z2of3',
                'z3of3')),
            partial(check_type, types=str),
            check_value_in_set,
            partial(check_key_in_dict, keys=(
                'viewpoint', 'time_variable', 'bounds')),
            partial(check_type, types=str),
            partial(check_value_in_set, options=(
                'early', 'late', 'long-term', 'longitudinal', 'profile')),
            partial(check_type, types=(type(None), str)),
            check_value_in_set,
            partial(check_type, types=list),
            partial(check_length, reference=2, sign='=='),
            partial(check_subtype, types=(type(None), int, float)))),
        ),
    'preprocessing_steps': (
        partial(check_type, types=list),
        partial(check_subtype, types=str),
        partial(check_value_in_set, options=tuple(
            {**cleaner_map, **reducer_map, **sampler_map, **transformer_map}))
        ),
    'architecture': (
        partial(check_type, types=str),
        partial(check_value_in_set, options=(
            'vanilla', 'vanilla-input-convex'))
        ),
    'max_hidden_layers': (
        partial(check_type, types=int),
        partial(check_value, reference=0, sign='>=')
        ),
    'tune_space': (
        partial(check_type, types=dict),
        ),
    'tune_evaluations': (
        partial(check_type, types=int),
        partial(check_value, reference=0, sign='>')
        ),
    'tune_score': (
        partial(check_type, types=str),
        partial(check_value_in_set, options=tuple(('AUC', *loss_map)))
        ),
    'tune_splits': (
        partial(check_type, types=int),
        partial(check_value, reference=1, sign='>')
        ),
    'inspect_model': (
        partial(check_type, types=bool),
        ),
    'evaluate_model': (
        partial(check_type, types=bool),
        ),
    'oof_splits': (
        partial(check_type, types=int),
        partial(check_value, reference=1, sign='>')
        ),
    'write_features': (
        partial(check_type, types=bool),
        ),
    'display_options': (
        partial(check_type, types=dict),
        partial(check_key_in_dict, keys=('graphs', 'kpis'))
        )
    }
