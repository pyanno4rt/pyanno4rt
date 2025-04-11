"""Optimization handler."""

# Author: Tim Ortkamp

# %% External package import

from copy import deepcopy
from functools import partial

# %% Internal package import

from pyanno4rt.checking import (
    check_length, check_subtype, check_type, check_value, check_value_in_set)
import pyanno4rt.optimization._maps as maps
from pyanno4rt.tools import filter_dict

# %% Class definition


class Optimization():
    """
    Optimization handler class.

    This class provides methods to handle the optimization parameters of a \
    treatment plan.

    Parameters
    ----------
    components : list
        Optimization components for each segment of interest, i.e., \
        objective functions and constraints.

        Currently available:

        - \
            :class:`~pyanno4rt.optimization.components._decision_tree_ntcp.DecisionTreeNTCP`
        - \
            :class:`~pyanno4rt.optimization.components._decision_tree_tcp.DecisionTreeTCP`
        - \
            :class:`~pyanno4rt.optimization.components._dose_uniformity.DoseUniformity`
        - \
            :class:`~pyanno4rt.optimization.components._equivalent_uniform_dose.EquivalentUniformDose`
        - \
            :class:`~pyanno4rt.optimization.components._k_nearest_neighbors_ntcp.KNeighborsNTCP`
        - \
            :class:`~pyanno4rt.optimization.components._k_nearest_neighbors_tcp.KNeighborsTCP`
        - \
            :class:`~pyanno4rt.optimization.components._logistic_regression_ntcp.LogisticRegressionNTCP`
        - \
            :class:`~pyanno4rt.optimization.components._logistic_regression_tcp.LogisticRegressionTCP`
        - \
            :class:`~pyanno4rt.optimization.components._lq_poisson_tcp.LQPoissonTCP`
        - \
            :class:`~pyanno4rt.optimization.components._lyman_kutcher_burman_ntcp.LymanKutcherBurmanNTCP`
        - :class:`~pyanno4rt.optimization.components._maximum_dvh.MaximumDVH`
        - :class:`~pyanno4rt.optimization.components._mean_dose.MeanDose`
        - :class:`~pyanno4rt.optimization.components._minimum_dvh.MinimumDVH`
        - \
            :class:`~pyanno4rt.optimization.components._naive_bayes_ntcp.NaiveBayesNTCP`
        - \
            :class:`~pyanno4rt.optimization.components._naive_bayes_tcp.NaiveBayesTCP`
        - \
            :class:`~pyanno4rt.optimization.components._neural_network_ntcp.NeuralNetworkNTCP`
        - \
            :class:`~pyanno4rt.optimization.components._neural_network_tcp.NeuralNetworkTCP`
        - \
            :class:`~pyanno4rt.optimization.components._random_forest_ntcp.RandomForestNTCP`
        - \
            :class:`~pyanno4rt.optimization.components._random_forest_tcp.RandomForestTCP`
        - \
            :class:`~pyanno4rt.optimization.components._squared_deviation.SquaredDeviation`
        - \
            :class:`~pyanno4rt.optimization.components._squared_overdosing.SquaredOverdosing`
        - \
            :class:`~pyanno4rt.optimization.components._squared_underdosing.SquaredUnderdosing`
        - \
            :class:`~pyanno4rt.optimization.components._support_vector_machine_ntcp.SupportVectorMachineNTCP`
        - \
            :class:`~pyanno4rt.optimization.components._support_vector_machine_tcp.SupportVectorMachineTCP`

        .. note:: To prevent overwriting processes, make use of the \
            identifier argument for components of the same type!

    method : {'lexicographic', 'pareto', 'weighted-sum'}, \
        default='weighted-sum'
        Single- or multi-criteria optimization method, see the classes \
        :class:`~pyanno4rt.optimization.methods._lexicographic_optimization.LexicographicOptimization`\
        :class:`~pyanno4rt.optimization.methods._pareto_optimization.ParetoOptimization`\
        :class:`~pyanno4rt.optimization.methods._weighted_sum_optimization.WeightedSumOptimization`.

        - 'lexicographic' : sequential optimization based on a preference \
            order
        - 'pareto' : parallel optimization based on the criterion of pareto \
            optimality
        - 'weighted-sum' : parallel optimization based on a weighted-sum \
            scalarization of the objective function

    solver : {'ipyopt', 'proxmin', 'pymoo', 'pypop7', 'scipy'}, default='scipy'
        Python package to be used for solving the optimization problem, \
        see the classes \
        :class:`~pyanno4rt.optimization.solvers._ipyopt_solver.IpyoptSolver`\
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

        - solver='ipyopt': {'mumps'}

            - 'mumps': multifrontal massively parallel sparse direct solver

        - solver='proxmin' : {'admm', 'pgm', 'sdmm'}

            - 'admm' : alternating direction method of multipliers
            - 'pgm' : proximal gradient method
            - 'sdmm' : simultaneous direction method of multipliers

        - solver='pymoo' : {'NSGA3'}

            - 'NSGA3' : non-dominated sorting genetic algorithm III

        - solver='pypop7' : {'LMCMA', 'LMMAES'}

            - 'LMCMA' : limited-memory covariance matrix adaptation
            - 'LMMAES' : limited-memory matrix adaptation evolution \
                strategy

        - solver='scipy' : {'L-BFGS-B', 'TNC', 'trust-constr'}

            - 'L-BFGS-B' : bounded limited memory \
                Broyden-Fletcher-Goldfarb-Shanno method
            - 'TNC' : truncated Newton method
            - 'trust-constr' : trust-region constrained method

        .. note:: Constraints are currently only supported by 'mumps', \
            'NSGA3' and 'trust-constr'.

    initial_strategy : {'data-medoid', 'target-coverage', 'warm-start'}, \
        default='target-coverage'
        Initialization strategy for the fluence vector, see the class \
        :class:`~pyanno4rt.optimization.initializers._fluence_initializer.FluenceInitializer`.

        - 'data-medoid' : fluence vector initialization with respect to data \
            medoid points
        - 'target-coverage' : fluence vector initialization with respect to \
            target coverage
        - 'warm-start' : fluence vector initialization with respect to a \
            reference optimal point

        .. note:: Data-medoid initialization works best for a single \
            dataset or multiple datasets with a high degree of \
            similarity. Otherwise, the data medoid point may lose its \
            individual representativeness.

    initial_fluence_vector : None or list, default=None
        Initial fluence vector for the optimization problem, only used if \
        initial_strategy='warm-start'.

    lower_variable_bounds : None, int, float, or list, default=0
        Lower bound(s) on the decision variables.

    upper_variable_bounds : None, int, float, or list, default=None
        Upper bound(s) on the decision variables.

    .. note:: There are two options to set lower and upper bounds for the \
        variables:

            1) Passing a single numeric value translates into uniform \
                bounds across all variables (where None for the lower \
                and/or upper bound indicates infinity bounds)
            2) Passing a list translates into non-uniform bounds (where the \
                length of the list must be equal to the number of decision \
                variables)

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

        # Get the input arguments
        inputs = filter_dict(vars(), remove_keys=('self',))

        # Check the input arguments
        self.check(inputs)

        # Loop over the input arguments
        for key, value in inputs.items():

            # Set the attribute
            setattr(self, key, value)

    def to_dict(self):
        """Serialize the object into a dictionary."""

        # Get the parameter dictionary
        dictionary = deepcopy(vars(self))

        # Serialize the components
        dictionary['components'] = [
            item.to_dict() for item in dictionary['components']]

        return dictionary

    @classmethod
    def from_dict(
            cls,
            dictionary):
        """
        Deserialize the object from a dictionary.

        Parameters
        ----------
        dictionary : dict
            Dictionary with the optimization parameters.

        Returns
        -------
        object of class :class:`~pyanno4rt.base._optimization.Optimization`
            The object used to handle the plan optimization parameters.
        """

        # Deserialize the components
        dictionary['components'] = [
            maps.COMPONENTS[key].from_dict(value)
            for item in dictionary['components']
            for key, value in item.items()]

        return cls(**dictionary)

    def check(
            self,
            inputs):
        """
        Check the input arguments.

        Parameters
        ----------
        inputs : dict
            Dictionary with the input arguments.
        """

        # Initialize the conditional variable dictionary
        conditions = {}

        # Loop over the conditional key-default pairs
        for key, default in (
                ('method', 'weighted-sum'),
                ('solver', 'scipy'),
                ('initial_strategy', 'target-coverage'),
                ('lower_variable_bounds', 0),
                ('upper_variable_bounds', None)):

            # Add the pair to the dictionary
            conditions[key] = inputs.get(key, getattr(self, key, default))

        # Get the check map
        check_map = {
            'components': (
                partial(check_type, types=list),
                partial(check_length, reference=1, sign='>='),
                partial(check_subtype, types=tuple(maps.COMPONENTS.values()))),
            'method': (
                partial(check_type, types=str),
                partial(check_value_in_set, options=tuple(maps.METHODS))),
            'solver': (
                partial(check_type, types=str),
                partial(check_value_in_set, options={
                    'lexicographic': ('scipy',),
                    'pareto': ('pymoo',),
                    'weighted-sum': ('ipyopt', 'proxmin', 'pypop7', 'scipy')},
                    value_condition=conditions['method'])),
            'algorithm': (
                partial(check_type, types=str),
                partial(check_value_in_set, options={
                    'weighted-sum/ipyopt': ('mumps',),
                    'weighted-sum/proxmin': ('admm', 'pgm', 'sdmm'),
                    'pareto/pymoo': ('NSGA3',),
                    'weighted-sum/pypop7': ('LMCMA', 'LMMAES'),
                    'lexicographic/scipy': ('trust-constr',),
                    'weighted-sum/scipy': ('L-BFGS-B', 'TNC', 'trust-constr')},
                    value_condition=(
                        f"{conditions['method']}/"
                        f"{conditions['solver']}"))),
            'initial_strategy': (
                partial(check_type, types=str),
                partial(check_value_in_set, options=(
                    'data-medoid', 'target-coverage', 'warm-start'))),
            'initial_fluence_vector': (
                partial(check_type, types={
                    'data-medoid': type(None),
                    'target-coverage': type(None),
                    'warm-start': list},
                    type_condition=conditions['initial_strategy']),
                partial(check_value, reference=0, sign='>=', is_vector=True)),
            'lower_variable_bounds': (
                partial(check_type, types=(type(None), int, float, list)),
                partial(check_value, reference=0, sign='>=',
                        is_vector=isinstance(
                            conditions['lower_variable_bounds'], list))),
            'upper_variable_bounds': (
                partial(check_type, types=(type(None), int, float, list)),
                partial(check_value, reference=0, sign='>=',
                        is_vector=isinstance(
                            conditions['upper_variable_bounds'], list))),
            'maximum_iterations': (
                partial(check_type, types=int),
                partial(check_value, reference=1, sign='>=')),
            'tolerance': (
                partial(check_type, types=float),
                partial(check_value, reference=0, sign='>'))}

        # Loop over the inputs
        for key, value in inputs.items():

            # Loop over the check functions
            for function in check_map[key]:

                # Run the check function
                function(key, value)
