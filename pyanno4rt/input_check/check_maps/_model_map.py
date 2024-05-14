"""Model parameter check map."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from functools import partial

# %% Internal package import

from pyanno4rt.input_check.check_functions import (
    check_regular_extension, check_feature_filter, check_key_in_dict,
    check_length, check_path, check_subtype, check_type, check_value,
    check_value_in_set)
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
        partial(check_type, types=str),
        partial(check_regular_extension, extensions=('.csv',)),
        ),
    'feature_filter': (
        partial(check_type, types=dict),
        partial(check_feature_filter, check_functions=(
            partial(check_key_in_dict, keys=('features', 'filter_mode')),
            partial(check_type, types=list),
            partial(check_subtype, types=str),
            partial(check_value_in_set, options=('remove', 'retain'))
            ))
        ),
    'label_name': (
        partial(check_type, types=str),
        ),
    'label_bounds': (
        partial(check_type, types=list),
        partial(check_length, reference=2, sign='=='),
        partial(check_subtype, types=(type(None), int, float)),
        partial(check_value, reference=0, sign='>=', is_vector=True)
        ),
    'time_variable_name': (
        partial(check_type, types={
            'early': str, 'late': str, 'long-term': str,
            'longitudinal': type(None), 'profile': str}),
        ),
    'label_viewpoint': (
        partial(check_type, types=str),
        partial(check_value_in_set, options=(
            'early', 'late', 'long-term', 'longitudinal', 'profile'))
        ),
    'fuzzy_matching': (
        partial(check_type, types=bool),
        ),
    'preprocessing_steps': (
        partial(check_type, types=list),
        partial(check_subtype, types=str),
        partial(check_value_in_set, options=tuple(
            {**cleaner_map, **reducer_map, **sampler_map, **transformer_map}))
        ),
    'architecture': (
        partial(check_type, types=str),
        partial(check_value_in_set, options=('input-convex', 'standard'))
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
