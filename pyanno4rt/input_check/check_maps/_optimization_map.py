"""Optimization check map."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from functools import partial

# %% Internal package import

from pyanno4rt.input_check.check_functions import (
    check_components, check_key_in_dict, check_length, check_subtype,
    check_type, check_value, check_value_in_set)

# %% Map definition


optimization_map = {
    'components': (
        partial(check_type, key_type=dict),
        partial(check_components, subfunctions=(
            partial(check_length, reference=0, sign='>'),
            partial(check_type, key_type=list),
            partial(check_length, reference=2, sign='=='),
            partial(check_value_in_set, options=('objective', 'constraint')),
            partial(check_type, key_type=(dict, list)),
            partial(check_subtype, key_type=dict),
            partial(check_key_in_dict, key_choices=('class', 'parameters')),
            partial(check_value_in_set, options=(
                'Decision Tree NTCP', 'Dose Uniformity',
                'Equivalent Uniform Dose',
                'Extreme Gradient Boosting NTCP',
                'K-Nearest Neighbors NTCP', 'Logistic Regression NTCP',
                'Logistic Regression TCP', 'Lyman-Kutcher-Burman NTCP',
                'Maximum DVH', 'Mean Dose', 'Minimum DVH', 'Moments',
                'Naive Bayes NTCP', 'Neural Network NTCP',
                'Neural Network TCP', 'Random Forest NTCP',
                'Squared Deviation', 'Squared Overdosing',
                'Squared Underdosing', 'Support Vector Machine NTCP',
                'Support Vector Machine TCP')),
            partial(check_type, key_type=dict)
            ))
        ),
    'method': (
        partial(check_type, key_type=str),
        partial(check_value_in_set, options=('pareto', 'weighted-sum'))
        ),
    'solver': (
        partial(check_type, key_type=str),
        partial(check_value_in_set, options={
            'pareto': ('pymoo',),
            'weighted-sum': ('ipopt', 'proxmin', 'pypop7', 'scipy')})
        ),
    'algorithm': (
        partial(check_type, key_type=str),
        partial(check_value_in_set, options={
            'ipopt': ('ma27', 'ma57', 'ma77', 'ma86'),
            'proxmin': ('admm', 'pgm', 'sdmm'),
            'pymoo': ('NSGA3',),
            'pypop7': ('MMES', 'LMCMA', 'RMES', 'BES', 'GS'),
            'scipy': ('L-BFGS-B', 'TNC', 'trust-constr')})
        ),
    'initial_strategy': (
        partial(check_type, key_type=str),
        partial(check_value_in_set, options=('target-coverage', 'warm-start'))
        ),
    'initial_fluence_vector': (
        partial(check_type, key_type={
            'target-coverage': type(None),
            'warm-start': list}),
        partial(check_value, reference=0, sign='>=', value_group='vector',
                type_group=list)
        ),
    'lower_variable_bounds': (
        partial(check_type, key_type=(int, float, type(None), list)),
        partial(check_value, reference=0, sign='>=', value_group='scalar'),
        partial(check_value, reference=0, sign='>=', value_group='vector')
        ),
    'upper_variable_bounds': (
        partial(check_type, key_type=(int, float, type(None), list)),
        partial(check_value, reference=0, sign='>=', value_group='scalar'),
        partial(check_value, reference=0, sign='>=', value_group='vector')
        ),
    'max_iter': (
        partial(check_type, key_type=int),
        partial(check_value, reference=0, sign='>', value_group='scalar')
        ),
    'max_cpu_time': (
        partial(check_type, key_type=float),
        partial(check_value, reference=0, sign='>', value_group='scalar')
        )
    }
