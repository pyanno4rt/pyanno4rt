"""Optimization handler."""

# Author: Tim Ortkamp

# %% External package import

from copy import deepcopy
from functools import partial

# %% Internal package import

import pyanno4rt.optimization._maps as maps
from pyanno4rt.tools import filter_dict
from pyanno4rt.validation import (
    validate_item, validate_item_in_set, validate_length, validate_subtype,
    validate_type)

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
            :class:`~pyanno4rt.optimization.components._decision_tree_outcome.DecisionTreeOutcome`
        - \
            :class:`~pyanno4rt.optimization.components._dose_uniformity.DoseUniformity`
        - \
            :class:`~pyanno4rt.optimization.components._equivalent_uniform_dose.EquivalentUniformDose`
        - \
            :class:`~pyanno4rt.optimization.components._k_nearest_neighbors_outcome.KNeighborsOutcome`
        - \
            :class:`~pyanno4rt.optimization.components._logistic_regression_outcome.LogisticRegressionOutcome`
        - \
            :class:`~pyanno4rt.optimization.components._lq_poisson_tcp.LQPoissonTCP`
        - \
            :class:`~pyanno4rt.optimization.components._lyman_kutcher_burman_ntcp.LymanKutcherBurmanNTCP`
        - :class:`~pyanno4rt.optimization.components._maximum_dvh.MaximumDVH`
        - :class:`~pyanno4rt.optimization.components._mean_dose.MeanDose`
        - :class:`~pyanno4rt.optimization.components._minimum_dvh.MinimumDVH`
        - \
            :class:`~pyanno4rt.optimization.components._naive_bayes_outcome.NaiveBayesOutcome`
        - \
            :class:`~pyanno4rt.optimization.components._neural_network_outcome.NeuralNetworkOutcome`
        - \
            :class:`~pyanno4rt.optimization.components._random_forest_outcome.RandomForestOutcome`
        - \
            :class:`~pyanno4rt.optimization.components._squared_deviation.SquaredDeviation`
        - \
            :class:`~pyanno4rt.optimization.components._squared_overdosing.SquaredOverdosing`
        - \
            :class:`~pyanno4rt.optimization.components._squared_underdosing.SquaredUnderdosing`
        - \
            :class:`~pyanno4rt.optimization.components._support_vector_machine_outcome.SupportVectorMachineOutcome`

    method : {'lexicographic', 'pareto', 'weighted-sum'}, \
        default='weighted-sum'
        Single- or multi-criteria optimization method, see the classes \
        :class:`~pyanno4rt.optimization.problems.lexicographic._lexicographic_problem.LexicographicProblem`\
        :class:`~pyanno4rt.optimization.problems.pareto._pareto_problem.ParetoProblem`\
        :class:`~pyanno4rt.optimization.problems.weighted._weighted_sum_problem.WeightedSumProblem`.

        - 'lexicographic' : sequential optimization based on a preference order
        - 'pareto' : parallel optimization based on the criterion of pareto \
            optimality
        - 'weighted-sum' : parallel optimization based on a weighted-sum \
            scalarization of the objective function

    solver : {'ipyopt', 'pyanno4rt', 'pymoo', 'pypop7', 'scipy'}, \
        default='scipy'
        Python package to be used for solving the optimization problem, see \
        the classes \
        :class:`~pyanno4rt.optimization.solvers._ipyopt_solver.IpyoptSolver`\
        :class:`~pyanno4rt.optimization.solvers._pyanno4rt_solver.Pyanno4rtSolver`\
        :class:`~pyanno4rt.optimization.solvers._pymoo_solver.PymooSolver`\
        :class:`~pyanno4rt.optimization.solvers._pypop7_solver.PyPop7Solver`\
        :class:`~pyanno4rt.optimization.solvers._scipy_solver.SciPySolver`.

        - 'ipyopt': interior-point algorithms provided by Ipyopt
        - 'pyanno4rt': internal custom algorithms provided by the package
        - 'pymoo' : multi-objective algorithms provided by Pymoo
        - 'pypop7': population-based algorithms provided by PyPop7
        - 'scipy' : local algorithms provided by SciPy

        .. note:: The 'lexicographic' method only works with 'ipyopt' and \
            'scipy', while the 'pareto' method only works with 'pymoo'.

    algorithm : str, default='L-BFGS-B'
        Solution algorithm from the chosen solver:

        - solver='ipyopt': {'mumps'}

            - 'mumps': multifrontal massively parallel sparse direct solver

        - solver='pyanno4rt': {'CMAES'}

            - 'CMAES': covariance matrix adaptation evolution strategy

        - solver='pymoo' : {'NSGA3'}

            - 'NSGA3' : non-dominated sorting genetic algorithm III

        - solver='pypop7' : {'LMCMA', 'LMMAES'}

            - 'LMCMA' : limited-memory covariance matrix adaptation
            - 'LMMAES' : limited-memory matrix adaptation evolution strategy

        - solver='scipy' : {'L-BFGS-B', 'TNC', 'trust-constr'}

            - 'L-BFGS-B' : bounded limited memory \
                Broyden-Fletcher-Goldfarb-Shanno method
            - 'TNC' : truncated Newton method
            - 'trust-constr' : trust-region constrained method

        .. note:: Constraints are currently only supported by 'mumps', \
            'NSGA3' and 'trust-constr'.

    initial_strategy : {'data-medoid', 'target-coverage', 'warm-start'}, \
        default='target-coverage'
        Initialization strategy for the fluence vector, see the classes \
        :class:`~pyanno4rt.optimization.initializers._data_medoid_initializer.DataMedoidInitializer`\
        :class:`~pyanno4rt.optimization.initializers._target_coverage_initializer.TargetCoverageInitializer`\
        :class:`~pyanno4rt.optimization.initializers._warm_start_initializer.WarmStartInitializer`.

        - 'data-medoid' : fluence vector initialization with respect to data \
            medoid points
        - 'target-coverage' : fluence vector initialization with respect to \
            target coverage
        - 'warm-start' : fluence vector initialization with respect to a \
            reference optimal point

        .. note:: Data-medoid initialization works best for a single \
            dataset or multiple datasets with high similarity. Otherwise, the \
            data medoid point may lack representativeness.

    initial_fluence : None or list, default=None
        Initial fluence vector for the optimization problem (only used if \
        initial_strategy='warm-start').

    lower_variable_bounds : None, int, float, or list, default=0
        Lower bound(s) on the decision variables.

    upper_variable_bounds : None, int, float, or list, default=None
        Upper bound(s) on the decision variables.

    .. note:: There are two options to set lower and upper variable bounds:

            1) Passing a single numeric value translates into uniform \
                bounds across all variables (where None for the lower \
                and/or upper bound indicates infinity bounds)
            2) Passing a list translates into non-uniform bounds (where the \
                length of the list must equal the number of decision variables)

    maximum_iterations : int, default=500
        Maximum number of iterations taken for the solver to converge. If set \
        to zero, the initial fluence is used as solution.

    tolerance : float, default=1e-3
        Precision goal for the objective function value.

    Attributes
    ----------
    components : list
        See 'Parameters'.

    method : {'lexicographic', 'pareto', 'weighted-sum'}
        See 'Parameters'.

    solver : {'ipyopt', 'pyanno4rt', 'pymoo', 'pypop7', 'scipy'}
        See 'Parameters'.

    algorithm : str
        See 'Parameters'.

    initial_strategy : {'data-medoid', 'target-coverage', 'warm-start'}
        See 'Parameters'.

    initial_fluence : None or list
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
            initial_fluence=None,
            lower_variable_bounds=0,
            upper_variable_bounds=None,
            maximum_iterations=500,
            tolerance=1e-3):

        # Validate the input arguments
        self.validate(filter_dict(vars(), remove_keys=('self',)))

        # Get the input attributes
        self.components = components
        self.method = method
        self.solver = solver
        self.algorithm = algorithm
        self.initial_strategy = initial_strategy
        self.initial_fluence = initial_fluence
        self.lower_variable_bounds = lower_variable_bounds
        self.upper_variable_bounds = upper_variable_bounds
        self.maximum_iterations = maximum_iterations
        self.tolerance = tolerance

    def to_dict(self):
        """Serialize the object into a dictionary."""

        # Get the parameter dictionary
        dictionary = deepcopy(vars(self))

        # Serialize the components
        dictionary['components'] = [
            component.to_dict() for component in dictionary['components']]

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
            for component in dictionary['components']
            for key, value in component.items()]

        return cls(**dictionary)

    def validate(
            self,
            inputs):
        """
        Validate the input arguments.

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
                ('upper_variable_bounds', None)
                ):

            # Add the pair to the dictionary
            conditions[key] = inputs.get(key, getattr(self, key, default))

        # Get the validation map
        validation_map = {
            'components': (
                partial(validate_type, options=list),
                partial(validate_length, reference=1, sign='>='),
                partial(
                    validate_subtype, options=(*maps.COMPONENTS.values(),))
                ),
            'method': (
                partial(validate_type, options=str),
                partial(validate_item_in_set, options=(*maps.PROBLEMS,))
                ),
            'solver': (
                partial(validate_type, options=str),
                partial(validate_item_in_set, options={
                    'lexicographic': ('ipyopt', 'scipy'),
                    'pareto': ('pymoo',),
                    'weighted-sum': (
                        'ipyopt', 'pyanno4rt', 'pypop7', 'scipy')},
                    condition=conditions['method'])
                ),
            'algorithm': (
                partial(validate_type, options=str),
                partial(validate_item_in_set, options={
                    'lexicographic/ipyopt': ('mumps',),
                    'weighted-sum/ipyopt': ('mumps',),
                    'weighted-sum/pyanno4rt': ('CMAES', 'LRCMAES'),
                    'pareto/pymoo': ('NSGA3',),
                    'weighted-sum/pypop7': ('LMCMA', 'LMMAES'),
                    'lexicographic/scipy': ('trust-constr',),
                    'weighted-sum/scipy': ('L-BFGS-B', 'TNC', 'trust-constr')},
                    condition=(
                        f"{conditions['method']}/"
                        f"{conditions['solver']}"))
                ),
            'initial_strategy': (
                partial(validate_type, options=str),
                partial(validate_item_in_set, options=(
                    'data-medoid', 'target-coverage', 'warm-start'))
                ),
            'initial_fluence': (
                partial(validate_type, options={
                    'data-medoid': (type(None), list),
                    'target-coverage': (type(None), list),
                    'warm-start': list},
                    condition=conditions['initial_strategy']),
                partial(validate_item, reference=0, sign='>=')
                ),
            'lower_variable_bounds': (
                partial(validate_type, options=(type(None), int, float, list)),
                partial(validate_item, reference=0, sign='>=')
                ),
            'upper_variable_bounds': (
                partial(validate_type, options=(type(None), int, float, list)),
                partial(validate_item, reference=0, sign='>=')
                ),
            'maximum_iterations': (
                partial(validate_type, options=int),
                partial(validate_item, reference=0, sign='>=')
                ),
            'tolerance': (
                partial(validate_type, options=(int, float)),
                partial(validate_item, reference=0, sign='>')
                )
            }

        # Loop over the inputs
        for key, value in inputs.items():

            # Loop over the validation functions
            for function in validation_map[key]:

                # Run the validation function
                function(key, value)
