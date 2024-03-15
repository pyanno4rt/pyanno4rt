"""Tune space check map."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from functools import partial

# %% Internal package import

from pyanno4rt.input_check.check_functions import (
    check_length, check_subtype, check_type, check_value, check_value_in_set)

# %% Map definition


tune_space_map = {
    'C': (
        partial(check_type, types=list),
        partial(check_length, reference=2, sign='=='),
        partial(check_subtype, types=(int, float)),
        partial(check_value, reference=0, sign='>', is_vector=True)
        ),
    'penalty': (
        partial(check_type, types=list),
        partial(check_value_in_set, options=('none', 'l1', 'l2'))
        ),
    'tol': (
        partial(check_type, types=list),
        partial(check_subtype, types=(int, float)),
        partial(check_value, reference=0, sign='>', is_vector=True)
        ),
    'class_weight': (
        partial(check_type, types=list),
        partial(check_value_in_set, options=(None, 'balanced'))
        ),
    'input_neuron_number': (
        partial(check_type, types=list),
        partial(check_subtype, types=int),
        partial(check_value, reference=0, sign='>', is_vector=True)
        ),
    'input_activation': (
        partial(check_type, types=list),
        partial(check_value_in_set, options=(
            'elu', 'gelu', 'leaky_relu', 'linear', 'relu', 'softmax',
            'softplus', 'swish'))
        ),
    'hidden_neuron_number': (
        partial(check_type, types=list),
        partial(check_subtype, types=int),
        partial(check_value, reference=0, sign='>', is_vector=True)
        ),
    'hidden_activation': (
        partial(check_type, types=list),
        partial(check_value_in_set, options=(
            'elu', 'gelu', 'leaky_relu', 'linear', 'relu', 'softmax',
            'softplus', 'swish'))
        ),
    'input_dropout_rate': (
        partial(check_type, types=list),
        partial(check_subtype, types=(int, float)),
        partial(check_value, reference=0, sign='>=', is_vector=True)
        ),
    'hidden_dropout_rate': (
        partial(check_type, types=list),
        partial(check_subtype, types=(int, float)),
        partial(check_value, reference=0, sign='>=', is_vector=True)
        ),
    'batch_size': (
        partial(check_type, types=list),
        partial(check_subtype, types=int),
        partial(check_value, reference=0, sign='>', is_vector=True)
        ),
    'learning_rate': (
        partial(check_type, types=list),
        partial(check_length, reference=2, sign='=='),
        partial(check_subtype, types=(int, float)),
        partial(check_value, reference=0, sign='>', is_vector=True)
        ),
    'optimizer': (
        partial(check_type, types=list),
        partial(check_value_in_set, options=('Adam', 'Ftrl', 'SGD'))
        ),
    'loss': (
        partial(check_type, types=list),
        partial(check_value_in_set, options=('BCE', 'FocalBCE', 'KLD'))
        ),
    'kernel': (
        partial(check_type, types=list),
        partial(check_value_in_set, options=(
            'linear', 'rbf', 'poly', 'sigmoid'))
        ),
    'degree': (
        partial(check_type, types=list),
        partial(check_subtype, types=int),
        partial(check_value, reference=0, sign='>', is_vector=True)
        ),
    'gamma': (
        partial(check_type, types=list),
        partial(check_length, reference=2, sign='=='),
        partial(check_subtype, types=(int, float)),
        partial(check_value, reference=0, sign='>', is_vector=True)
        )
    }
