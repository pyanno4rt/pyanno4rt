"""Model parameter check map."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from functools import partial

# %% Internal package import

from pyanno4rt.input_check.check_functions import (
    check_extension, check_feature_filter, check_length, check_subtype,
    check_type, check_value, check_value_in_set)

# %% Map definition


model_map = {
    'model_label': (
        partial(check_type, key_type=str),
        ),
    'model_folder_path': (
        partial(check_type, key_type=(type(None), str)),
        ),
    'data_path': (
        partial(check_type, key_type=str),
        partial(check_extension, extensions=('.csv',))
        ),
    'feature_filter': (
        partial(check_type, key_type=list),
        partial(check_length, reference=2, sign='=='),
        partial(check_feature_filter, subfunctions=(
            partial(check_type, key_type=list),
            partial(check_subtype, key_type=str),
            partial(check_type, key_type=str),
            partial(check_value_in_set, options=('remove', 'retain'))
            ))
        ),
    'label_viewpoint': (
        partial(check_type, key_type=str),
        partial(check_value_in_set, options=('early', 'late',
                                             'long-term',
                                             'longitudinal',
                                             'profile'))
        ),
    'label_bounds': (
        partial(check_type, key_type=list),
        partial(check_length, reference=2, sign='=='),
        partial(check_subtype, key_type=(type(None), int, float)),
        partial(check_value, reference=0, sign='>=',
                value_group='vector')
        ),
    'fuzzy_matching': (
        partial(check_type, key_type=bool),
        ),
    'preprocessing_steps': (
        partial(check_type, key_type=list),
        partial(check_subtype, key_type=str)
        ),
    'architecture': (
        partial(check_type, key_type=str),
        partial(check_value_in_set, options=('input-convex',
                                             'standard'))
        ),
    'max_hidden_layers': (
        partial(check_type, key_type=int),
        partial(check_value, reference=0, sign='>=',
                value_group='scalar')
        ),
    'tune_space': (
        partial(check_type, key_type=dict),
        ),
    'tune_evaluations': (
        partial(check_type, key_type=int),
        partial(check_value, reference=0, sign='>',
                value_group='scalar')
        ),
    'tune_score': (
        partial(check_type, key_type=str),
        partial(check_value_in_set, options=('Logloss',
                                             'Brier score',
                                             'AUC'))
        ),
    'tune_splits': (
        partial(check_type, key_type=int),
        partial(check_value, reference=1, sign='>',
                value_group='scalar')
        ),
    'inspect_model': (
        partial(check_type, key_type=bool),
        ),
    'evaluate_model': (
        partial(check_type, key_type=bool),
        ),
    'oof_splits': (
        partial(check_type, key_type=int),
        partial(check_value, reference=1, sign='>',
                value_group='scalar')
        ),
    'write_features': (
        partial(check_type, key_type=bool),
        ),
    'write_gradients': (
        partial(check_type, key_type=bool),
        ),
    'display_options': (
        partial(check_type, key_type=dict),
        )
    }
