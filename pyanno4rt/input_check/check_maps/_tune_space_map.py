"""Tune space check map."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from functools import partial

# %% Internal package import

from pyanno4rt.input_check.check_functions import (
    check_length, check_subtype, check_type, check_value, check_value_in_set)
from pyanno4rt.learning_model.frequentist.addons import loss_map, optimizer_map

# %% Map definition


tune_space_map = {
    'criterion': (
        partial(check_type, types=list),
        partial(check_value_in_set, options=('gini', 'entropy'))
        ),
    'splitter': (
        partial(check_type, types=list),
        partial(check_value_in_set, options=('best', 'random'))
        ),
    'max_depth': (
        partial(check_type, types=list),
        partial(check_subtype, types=int),
        partial(check_value, reference=0, sign='>', is_vector=True)
        ),
    'min_samples_split': (
        partial(check_type, types=list),
        partial(check_length, reference=2, sign='=='),
        partial(check_subtype, types=float),
        partial(check_value, reference=0, sign='>=', is_vector=True),
        partial(check_value, reference=1, sign='<=', is_vector=True)
        ),
    'min_samples_leaf': (
        partial(check_type, types=list),
        partial(check_length, reference=2, sign='=='),
        partial(check_subtype, types=float),
        partial(check_value, reference=0, sign='>=', is_vector=True),
        partial(check_value, reference=1, sign='<=', is_vector=True)
        ),
    'min_weight_fraction_leaf': (
        partial(check_type, types=list),
        partial(check_length, reference=2, sign='=='),
        partial(check_subtype, types=float),
        partial(check_value, reference=0, sign='>=', is_vector=True),
        partial(check_value, reference=1, sign='<=', is_vector=True)
        ),
    'max_features': (
        partial(check_type, types=list),
        partial(check_subtype, types=int),
        partial(check_value, reference=0, sign='>', is_vector=True)
        ),
    'class_weight': (
        partial(check_type, types=list),
        partial(check_value_in_set, options=(None, 'balanced'))
        ),
    'ccp_alpha': (
        partial(check_type, types=list),
        partial(check_length, reference=2, sign='=='),
        partial(check_subtype, types=float),
        partial(check_value, reference=0, sign='>=', is_vector=True),
        partial(check_value, reference=1, sign='<=', is_vector=True)
        ),
    'n_neighbors': (
        partial(check_type, types=list),
        partial(check_subtype, types=int),
        partial(check_value, reference=0, sign='>', is_vector=True)
        ),
    'weights': (
        partial(check_type, types=list),
        partial(check_value_in_set, options=('uniform', 'distance'))
        ),
    'leaf_size': (
        partial(check_type, types=list),
        partial(check_subtype, types=int),
        partial(check_value, reference=0, sign='>', is_vector=True)
        ),
    'p': (
        partial(check_type, types=list),
        partial(check_subtype, types=int),
        partial(check_value, reference=0, sign='>', is_vector=True)
        ),
    'C': (
        partial(check_type, types=list),
        partial(check_length, reference=2, sign='=='),
        partial(check_subtype, types=(int, float)),
        partial(check_value, reference=0, sign='>', is_vector=True)
        ),
    'penalty': (
        partial(check_type, types=list),
        partial(check_value_in_set, options=('l1', 'l2'))
        ),
    'tol': (
        partial(check_type, types=list),
        partial(check_subtype, types=(int, float)),
        partial(check_value, reference=0, sign='>', is_vector=True)
        ),
    'priors': (
        partial(check_type, types=list),
        partial(check_subtype, types=(list, type(None)))
        ),
    'var_smoothing': (
        partial(check_type, types=list),
        partial(check_length, reference=2, sign='=='),
        partial(check_subtype, types=(int, float)),
        partial(check_value, reference=0, sign='>', is_vector=True)
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
        partial(check_value_in_set, options=tuple(optimizer_map))
        ),
    'loss': (
        partial(check_type, types=list),
        partial(check_value_in_set, options=tuple(loss_map))
        ),
    'n_estimators': (
        partial(check_type, types=list),
        partial(check_subtype, types=int),
        partial(check_value, reference=0, sign='>', is_vector=True)
        ),
    'bootstrap': (
        partial(check_type, types=list),
        partial(check_value_in_set, options=(False, True))
        ),
    'warm-start': (
        partial(check_type, types=list),
        partial(check_value_in_set, options=(False, True))
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
