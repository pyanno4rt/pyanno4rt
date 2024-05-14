"""Pymoo algorithm configuration."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>
# Reference: https://pymoo.org/

# %% External package import

from numpy import array
from numpy.random import beta
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.population import Population
from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.util.ref_dirs import get_reference_directions

# %% Function definition


def configure_pymoo(number_of_variables, number_of_objectives,
                    number_of_constraints, problem_instance,
                    lower_variable_bounds, upper_variable_bounds,
                    lower_constraint_bounds, upper_constraint_bounds,
                    algorithm, initial_fluence, max_iter, tolerance):
    """
    Configure the Pymoo solver.

    Supported algorithms: NSGA-3.

    Parameters
    ----------
    number_of_variables : int
        Number of decision variables.

    number_of_objectives : int
        Number of objective functions.

    number_of_constraints : int
        Number of constraint functions.

    problem_instance : object of class \
        :class:`~pyanno4rt.optimization.methods._pareto_optimization.ParetoOptimization`\
        The object representing the optimization problem.

    lower_variable_bounds : list
        Lower bounds on the decision variables.

    upper_variable_bounds : list
        Upper bounds on the decision variables.

    lower_constraint_bounds : list
        Lower bounds on the constraints.

    upper_constraint_bounds : list
        Upper bounds on the constraints.

    algorithm : str
        Label for the solution algorithm.

    initial_fluence : ndarray
        Initial fluence vector.

    max_iter : int
        Maximum number of iterations.

    tolerance : float
        Precision goal for the objective function value.

    Returns
    -------
    fun : callable
        Minimization function from the Pymoo library.

    algorithm_object : object of class from :mod:`pymoo.algorithms`
        The object representing the solution algorithm.

    problem : object of class from :mod:`pymoo.core.problem`
        The object representing the Pymoo-compatible structure of the \
        multi-objective (Pareto) optimization problem.

    termination : object of class from :mod:`pymoo.termination`
        The object representing the termination criterion.
    """

    # Set the optimization function
    fun = minimize

    # Initialize the Pymoo problem instance
    problem = PymooProblem(
        number_of_variables=number_of_variables,
        number_of_objectives=number_of_objectives,
        number_of_constraints=number_of_constraints,
        problem_instance=problem_instance,
        lower_variable_bounds=lower_variable_bounds,
        upper_variable_bounds=[1e12]*len(upper_variable_bounds),
        lower_constraint_bounds=lower_constraint_bounds,
        upper_constraint_bounds=upper_constraint_bounds)

    # Check if the algorithm is 'NSGA3'
    if algorithm == 'NSGA3':

        # Set the number of points to evaluate
        number_of_points = 200

        # Get the reference directions
        reference_directions = get_reference_directions(
            "energy", number_of_objectives, number_of_points, seed=1)

        # Initialize and evaluate the initial population
        initial_population = Population.new(
            "X", 2*max(initial_fluence)*beta(
                a=0.5, b=0.5, size=(number_of_points, number_of_variables)))

        # Initialize the NSGA-3 algorithm
        algorithm_object = NSGA3(
            ref_dirs=reference_directions,
            pop_size=number_of_points,
            n_offsprings=number_of_points,
            sampling=initial_population,
            crossover=UniformCrossover(prob=1.0),
            mutation=PM(prob=1/number_of_variables, eta=20),
            eliminate_duplicates=True)

    # Initialize the termination instance
    termination = DefaultMultiObjectiveTermination(
        xtol=1e-12, ftol=tolerance, n_max_gen=max_iter)

    return fun, algorithm_object, problem, termination


class PymooProblem(ElementwiseProblem):
    """
    Pymoo optimization problem class.

    This class provides a Pymoo-compatible structure of the problem and a \
    method to evaluate the fitness of the solution in each iteration.

    Parameters
    ----------
    number_of_variables : int
        Number of decision variables.

    number_of_objectives : int
        Number of objective functions.

    number_of_constraints : int
        Number of constraint functions.

    problem_instance : object of class \
        :class:`~pyanno4rt.optimization.methods._pareto_optimization.ParetoOptimization`\
        The object representing the optimization problem.

    lower_variable_bounds : list
        Lower bounds on the decision variables.

    upper_variable_bounds : list
        Upper bounds on the decision variables.

    lower_constraint_bounds : list
        Lower bounds on the constraints.

    upper_constraint_bounds : list
        Upper bounds on the constraints.

    Attributes
    ----------
    problem_instance : object of class \
        :class:`~pyanno4rt.optimization.methods._pareto_optimization.ParetoOptimization`
        See 'Parameters'.

    Notes
    -----
    Fitness evaluation is based on a modification suggested in the paper by \
    Pang et al. (2020): DOI 10.1109/ACCESS.2020.3032240.
    """

    def __init__(
            self,
            number_of_variables,
            number_of_objectives,
            number_of_constraints,
            problem_instance,
            lower_variable_bounds,
            upper_variable_bounds,
            lower_constraint_bounds,
            upper_constraint_bounds):

        # Call the superclass constructor
        super().__init__(n_var=number_of_variables,
                         n_obj=number_of_objectives,
                         n_ieq_constr=2*number_of_constraints,
                         xl=array(lower_variable_bounds),
                         xu=array(upper_variable_bounds))

        # Get the instance attributes from the argument
        self.problem_instance = problem_instance
        self.lower_constraint_bounds = lower_constraint_bounds
        self.upper_constraint_bounds = upper_constraint_bounds

    def _evaluate(
            self,
            x,
            out,
            *args,
            **kwargs):
        """
        Evaluate the set of objective functions.

        Parameters
        ----------
        x : ndarray
            Current decision vector.

        out : dict
            Dictionary with the objective and constraint values.

        *args : tuple
            Tuple with optional (non-keyworded) parameters.

        **kwargs : dict
            Dictionary with optional (keyworded) parameters.
        """

        # Get the objective function value(s)
        objective_values = self.problem_instance.objective(x)

        # Set the mixture parameter
        alpha = 0.5

        # Add the mixture fitness values to the output dictionary
        out['F'] = [(1-alpha)*value
                    + (alpha/len(objective_values))*sum(objective_values)
                    for value in objective_values]

        # Get the constraint function value(s)
        constraint_values = self.problem_instance.constraint(x)

        # Add the constraint values to the output dictionary
        out['G'] = [[self.lower_constraint_bounds[index]-value,
                     value-self.upper_constraint_bounds[index]]
                    for index, value in enumerate(constraint_values)]
