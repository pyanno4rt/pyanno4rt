"""Optimization check map."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from functools import partial

# %% Internal package import

from pyanno4rt.input_check.check_functions import (
    check_components, check_key_in_dict, check_subtype, check_type,
    check_value, check_value_in_set)
from pyanno4rt.optimization.components import component_map
from pyanno4rt.optimization.methods import method_map

# %% Map definition


optimization_map = {
    'components': (
        partial(check_type, types=dict),
        partial(check_components, check_functions=(
            partial(check_key_in_dict, keys=('type', 'instance')),
            partial(check_value_in_set, options=('objective', 'constraint')),
            partial(check_type, types=(dict, list)),
            partial(check_key_in_dict, keys=('class', 'parameters')),
            partial(check_value_in_set, options=tuple(component_map)),
            partial(check_type, types=dict),
            partial(check_subtype, types=dict)
            ))
        ),
    'method': (
        partial(check_type, types=str),
        partial(check_value_in_set, options=tuple(method_map))
        ),
    'solver': (
        partial(check_type, types=str),
        partial(check_value_in_set, options={
            'lexicographic': ('', 'scipy'),
            'pareto': ('pymoo',),
            'weighted-sum': ('proxmin', 'pypop7', 'scipy')})
        ),
    'algorithm': (
        partial(check_type, types=str),
        partial(check_value_in_set, options={
            'lexicographic/proxmin': (),
            'weighted-sum/proxmin': ('admm', 'pgm', 'sdmm'),
            'pareto/pymoo': ('NSGA3',),
            'lexicographic/pypop7': (),
            'weighted-sum/pypop7': ('LMCMA', 'LMMAES'),
            'lexicographic/scipy': ('trust-constr',),
            'weighted-sum/scipy': ('L-BFGS-B', 'TNC', 'trust-constr')})
        ),
    'initial_strategy': (
        partial(check_type, types=str),
        partial(check_value_in_set, options=(
            'data-medoid', 'target-coverage', 'warm-start'))
        ),
    'initial_fluence_vector': (
        partial(check_type, types={
            'data-medoid': type(None),
            'target-coverage': type(None),
            'warm-start': list}),
        partial(check_value, reference=0, sign='>=', is_vector=True)
        ),
    'lower_variable_bounds': (
        partial(check_type, types=(type(None), int, float, list)),
        partial(check_value, reference=0, sign='>='),
        ),
    'upper_variable_bounds': (
        partial(check_type, types=(type(None), int, float, list)),
        partial(check_value, reference=0, sign='>='),
        ),
    'max_iter': (
        partial(check_type, types=int),
        partial(check_value, reference=0, sign='>')
        ),
    'tolerance': (
        partial(check_type, types=float),
        partial(check_value, reference=0, sign='>')
        )
    }
