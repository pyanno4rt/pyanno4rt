"""Model parameter check map."""

# Author: Tim Ortkamp

# %% External package import

from functools import partial

# %% Internal package import

from pyanno4rt.input_check.check_functions import (
    check_data_columns, check_key_in_dict, check_path, check_regular_extension,
    check_regular_extension_directory, check_subtype, check_type, check_value,
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
        partial(check_type, types={True: (type(None), str), False: str}),
        partial(check_regular_extension, extensions=('.csv',)),
        partial(check_regular_extension_directory, extensions=(
            'jpg', 'npy', 'npz', 'png'), no_directory=('.csv',))
        ),
    'data_columns': (
        partial(check_type, types=dict),
        partial(check_data_columns, ())
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
