"""Plan optimization information."""

# Author: Tim Ortkamp

# %% External package import

from functools import partial

# %% Internal package import

from pyanno4rt.input_check.check_functions import (
    check_subtype, check_type, check_value, check_value_in_set)
from pyanno4rt.optimization.components import component_map
from pyanno4rt.optimization.methods import method_map
from pyanno4rt.tools import filter_dict

# %% Class definition


class Optimization():
    """
    Plan optimization information class.

    This class provides methods to set, validate and serialize the \
    optimization parameters of the treatment plan.

    Parameters
    ----------
    components : list
        Optimization components for each segment of interest, i.e., \
        objective functions and constraints.

        Currently available:

        - :class:`~pyanno4rt.optimization.components.\
            _decision_tree_ntcp.DecisionTreeNTCP`
        - :class:`~pyanno4rt.optimization.components.\
            _decision_tree_tcp.DecisionTreeTCP`
        - :class:`~pyanno4rt.optimization.components.\
            _dose_uniformity.DoseUniformity`
        - :class:`~pyanno4rt.optimization.components.\
            _equivalent_uniform_dose.EquivalentUniformDose`
        - :class:`~pyanno4rt.optimization.components.\
            _k_nearest_neighbors_ntcp.KNeighborsNTCP`
        - :class:`~pyanno4rt.optimization.components.\
            _k_nearest_neighbors_tcp.KNeighborsTCP`
        - :class:`~pyanno4rt.optimization.components.\
            _logistic_regression_ntcp.LogisticRegressionNTCP`
        - :class:`~pyanno4rt.optimization.components.\
            _logistic_regression_tcp.LogisticRegressionTCP`
        - :class:`~pyanno4rt.optimization.components.\
            _lq_poisson_tcp.LQPoissonTCP`
        - :class:`~pyanno4rt.optimization.components.\
            _lyman_kutcher_burman_ntcp.LymanKutcherBurmanNTCP`
        - :class:`~pyanno4rt.optimization.components.\
            _maximum_dvh.MaximumDVH`
        - :class:`~pyanno4rt.optimization.components.\
            _mean_dose.MeanDose`
        - :class:`~pyanno4rt.optimization.components.\
            _minimum_dvh.MinimumDVH`
        - :class:`~pyanno4rt.optimization.components.\
            _naive_bayes_ntcp.NaiveBayesNTCP`
        - :class:`~pyanno4rt.optimization.components.\
            _naive_bayes_tcp.NaiveBayesTCP`
        - :class:`~pyanno4rt.optimization.components.\
            _neural_network_ntcp.NeuralNetworkNTCP`
        - :class:`~pyanno4rt.optimization.components.\
            _neural_network_tcp.NeuralNetworkTCP`
        - :class:`~pyanno4rt.optimization.components.\
            _random_forest_ntcp.RandomForestNTCP`
        - :class:`~pyanno4rt.optimization.components.\
            _random_forest_tcp.RandomForestTCP`
        - :class:`~pyanno4rt.optimization.components.\
            _squared_deviation.SquaredDeviation`
        - :class:`~pyanno4rt.optimization.components.\
            _squared_overdosing.SquaredOverdosing`
        - :class:`~pyanno4rt.optimization.components.\
            _squared_underdosing.SquaredUnderdosing`
        - :class:`~pyanno4rt.optimization.components.\
            _support_vector_machine_ntcp.SupportVectorMachineNTCP`
        - :class:`~pyanno4rt.optimization.components.\
            _support_vector_machine_tcp.SupportVectorMachineTCP`

        .. note:: If you attempt to set multiple objectives or multiple \
            constraints for a single segment, you should make use of the \
            identifier argument of the components!

    method : {'lexicographic', 'pareto', 'weighted-sum'}, \
        default='weighted-sum'
        Single- or multi-criteria optimization method, see the classes \
        :class:`~pyanno4rt.optimization.methods._lexicographic_optimization.LexicographicOptimization`\
        :class:`~pyanno4rt.optimization.methods._pareto_optimization.ParetoOptimization`\
        :class:`~pyanno4rt.optimization.methods._weighted_sum_optimization.WeightedSumOptimization`.

        - 'lexicographic' : sequential optimization based on a \
            preference order
        - 'pareto' : parallel optimization based on the criterion of \
            pareto optimality
        - 'weighted-sum' : parallel optimization based on a weighted-sum \
            scalarization of the objective function

    solver : {'ipyopt', 'proxmin', 'pymoo', 'pypop7', 'scipy'}, default='scipy'
        Python package to be used for solving the optimization problem, \
        see the classes \
        :class:`~pyanno4rt.optimization.solvers._proxmin_solver.IpyoptSolver`\
        :class:`~pyanno4rt.optimization.solvers._proxmin_solver.ProxminSolver`\
        :class:`~pyanno4rt.optimization.solvers._pymoo_solver.PymooSolver`\
        :class:`~pyanno4rt.optimization.solvers._pypop7_solver.PyPop7Solver`\
        :class:`~pyanno4rt.optimization.solvers._scipy_solver.SciPySolver`.

        - 'ipyopt': interior-point algorithms provided by Ipyopt
        - 'proxmin' : proximal algorithms provided by Proxmin
        - 'pymoo' : multi-objective algorithms provided by Pymoo
        - 'pypop7': population-based algorithms provided by PyPop7
        - 'scipy' : local algorithms provided by SciPy

        .. note:: The 'lexicographic' method currently only works with \
            'scipy', while the 'pareto' method only works with 'pymoo'.

    algorithm : str, default='L-BFGS-B'
        Solution algorithm from the chosen solver:

        - solver='ipyopt': {'mumps'}, default='mumps'

            - 'mumps': multifrontal massively parallel sparse direct solver

        - solver='proxmin' : {'admm', 'pgm', 'sdmm'}, default='pgm'

            - 'admm' : alternating direction method of multipliers
            - 'pgm' : proximal gradient method
            - 'sdmm' : simultaneous direction method of multipliers

        - solver='pymoo' : {'NSGA3'}, default='NSGA3'

            - 'NSGA3' : non-dominated sorting genetic algorithm III

        - solver='pypop7' : {'LMCMA', 'LMMAES'}, default='LMCMA'

            - 'LMCMA' : limited-memory covariance matrix adaptation
            - 'LMMAES' : limited-memory matrix adaptation evolution \
                strategy

        - solver='scipy' : {'L-BFGS-B', 'TNC', 'trust-constr'}, \
            default='L-BFGS-B'

            - 'L-BFGS-B' : bounded limited memory \
                Broyden-Fletcher-Goldfarb-Shanno method
            - 'TNC' : truncated Newton method
            - 'trust-constr' : trust-region constrained method

        .. note:: Constraints are currently only supported by 'mumps', \
            'NSGA3' and 'trust-constr'.

    initial_strategy : {'data-medoid', 'target-coverage', 'warm-start'}, \
        default='target-coverage'
        Initialization strategy for the fluence vector (see the class \
        :class:`~pyanno4rt.optimization.initializers._fluence_initializer.FluenceInitializer`).

        - 'data-medoid' : fluence vector initialization with respect to \
            data medoid points
        - 'target-coverage' : fluence vector initialization with respect \
            to tumor coverage
        - 'warm-start' : fluence vector initialization with respect to a \
            reference optimal point

        .. note:: Data-medoid initialization works best for a single \
            dataset or multiple datasets with a high degree of \
            similarity. Otherwise, the initial fluence vector may lose \
            its individual representativeness.

    initial_fluence_vector : None or list, default=None
        User-defined initial fluence vector for the optimization problem, \
        only used if initial_strategy='warm-start' (see the class \
        :class:`~pyanno4rt.optimization.initializers._fluence_initializer.FluenceInitializer`).

    lower_variable_bounds : None, int, float, or list, default=0
        Lower bound(s) on the decision variables.

    upper_variable_bounds : None, int, float, or list, default=None
        Upper bound(s) on the decision variables.

    .. note:: There are two options to set lower and upper bounds for the \
        variables:

            1) Passing a single numeric value translates into uniform \
                bounds across all variables (where None for the lower \
                and/or upper bound indicates infinity bounds)
            2) Passing a list translates into non-uniform bounds (here, \
                the length of the list needs to be equal to the number of \
                decision variables)

    maximum_iterations : int, default=500
        Maximum number of iterations taken for the solver to converge.

    tolerance : float, default=1e-3
        Precision goal for the objective function value.

    Attributes
    ----------
    components : list
        See 'Parameters'.

    method : {'lexicographic', 'pareto', 'weighted-sum'}
        See 'Parameters'.

    solver : {'ipyopt', 'proxmin', 'pymoo', 'pypop7', 'scipy'}
        See 'Parameters'.

    algorithm : str
        See 'Parameters'.

    initial_strategy : {'data-medoid', 'target-coverage', 'warm-start'}
        See 'Parameters'.

    initial_fluence_vector : None or list
        See 'Parameters'.

    lower_variable_bounds : None, int, float, or list
        See 'Parameters'.

    upper_variable_bounds : None, int, float, or list
        See 'Parameters'.

    maximum_iterations : int
        See 'Parameters'.

    tolerance : float
        See 'Parameters'.
    """

    def __init__(
            self,
            components,
            method='weighted-sum',
            solver='scipy',
            algorithm='L-BFGS-B',
            initial_strategy='target-coverage',
            initial_fluence_vector=None,
            lower_variable_bounds=0,
            upper_variable_bounds=None,
            maximum_iterations=500,
            tolerance=1e-3):

        # Check the input arguments
        self.check(filter_dict(vars(), remove_keys=('self',)))

        # Loop over the input arguments
        for key, value in filter_dict(vars(), remove_keys=('self',)).items():

            # Set the attribute
            setattr(self, key, value)

    def to_dict(self):
        """Return the attribute dictionary."""

        return vars(self)

    def check(
            self,
            input_dictionary):
        """
        Check the items of an input dictionary.

        Parameters
        ----------
        input_dictionary : dict
            Dictionary with the mappings between parameter names and values.
        """

        # Get the check map
        check_map = {
            'components': (
                partial(check_type, types=list),
                partial(check_subtype, types=tuple(component_map.values()))),
            'method': (
                partial(check_type, types=str),
                partial(check_value_in_set, options=tuple(method_map))),
            'solver': (
                partial(check_type, types=str),
                partial(check_value_in_set, options={
                    'lexicographic': ('scipy',),
                    'pareto': ('pymoo',),
                    'weighted-sum': ('ipyopt', 'proxmin', 'pypop7', 'scipy')})
                ),
            'algorithm': (
                partial(check_type, types=str),
                partial(check_value_in_set, options={
                    'weighted-sum + ipyopt': ('mumps',),
                    'weighted-sum + proxmin': ('admm', 'pgm', 'sdmm'),
                    'pareto + pymoo': ('NSGA3',),
                    'weighted-sum + pypop7': ('LMCMA', 'LMMAES'),
                    'lexicographic + scipy': ('trust-constr',),
                    'weighted-sum + scipy': ('L-BFGS-B', 'TNC', 'trust-constr')
                    })),
            'initial_strategy': (
                partial(check_type, types=str),
                partial(check_value_in_set, options=(
                    'data-medoid', 'target-coverage', 'warm-start'))),
            'initial_fluence_vector': (
                partial(check_type, types={
                    'data-medoid': type(None),
                    'target-coverage': type(None),
                    'warm-start': list}),
                partial(check_value, reference=0, sign='>=', is_vector=True)),
            'lower_variable_bounds': (
                partial(check_type, types=(type(None), int, float, list)),
                partial(check_value, reference=0, sign='>=')),
            'upper_variable_bounds': (
                partial(check_type, types=(type(None), int, float, list)),
                partial(check_value, reference=0, sign='>=')),
            'maximum_iterations': (
                partial(check_type, types=int),
                partial(check_value, reference=0, sign='>')),
            'tolerance': (
                partial(check_type, types=float),
                partial(check_value, reference=0, sign='>'))}

        # Set the additional check function arguments
        args = {
            'solver': {
                'value_condition': input_dictionary.get('method')},
            'algorithm': {
                'value_condition': (
                    f"{input_dictionary.get('method')} + "
                    f"{input_dictionary.get('solver')}")},
            'initial_fluence_vector': {
                'type_condition': input_dictionary.get('initial_strategy')}}

        # Loop over the dictionary keys
        for key, value in input_dictionary.items():

            # Check if the key is included in the check map
            if key in check_map:

                # Check if the key holds vector-like lower or upper bounds
                if (key in ('lower_variable_bounds', 'upper_variable_bounds')
                        and not isinstance(value, (int, float, type(None)))):

                    # Add the corresponding additional argument
                    args[key] = {'is_vector': True}

                # Loop over the check functions
                for function in check_map[key]:

                    # Get the additional arguments
                    key_args = args.get(key, {})

                    # Get the function arguments
                    func_args = function.func.__code__.co_varnames

                    # Get the additional arguments filtered by function
                    filter_args = {
                        arg: key_args[arg] for arg in func_args
                        if arg in key_args}

                    # Run the check function
                    function(key, value, **filter_args)
