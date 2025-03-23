"""Pymoo wrapper."""

# Author: Tim Ortkamp
# Reference: https://pymoo.org/

# %% External package import

from numpy import array, mean
from numpy.random import beta
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.callback import Callback
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.population import Population
from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.util.ref_dirs import get_reference_directions

# %% Internal package import

from pyanno4rt.datahub import Datahub
from pyanno4rt.tools import (
    get_all_constraints, get_all_objectives, get_constraint_segments,
    get_objective_segments)

# %% Class definition


class PymooSolver():
    """
    Pymoo wrapper class.

    This class serves as a wrapper for the multi-objective (Pareto) \
    optimization algorithms from the Pymoo solver. It takes the problem \
    structure, configures the selected algorithm, and defines the method to \
    run the solver.

    Parameters
    ----------
    number_of_variables : int
        Number of decision variables.

    number_of_constraints : int
        Number of constraints.

    problem_instance : object of class \
        :class:`~pyanno4rt.optimization.methods._pareto_optimization.ParetoOptimization`\
        The object representing the (Pareto) optimization problem.

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

    maximum_iterations : int
        Maximum number of iterations.

    tolerance : float
        Precision goal for the objective function value.

    Attributes
    ----------
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

    def __init__(
            self,
            number_of_variables,
            number_of_constraints,
            problem_instance,
            lower_variable_bounds,
            upper_variable_bounds,
            lower_constraint_bounds,
            upper_constraint_bounds,
            algorithm,
            initial_fluence,
            maximum_iterations,
            tolerance):

        # Initialize the datahub
        hub = Datahub()

        # Log a message about the initialization of the class
        hub.logger.display_info(
            f"Initializing Pymoo solver with {algorithm} algorithm ...")

        # Get the callable optimization function and the solver objects
        self.fun, self.algorithm_object, self.problem, self.termination = (
            self.configure(
                number_of_variables, len(get_all_objectives(hub.segmentation)),
                number_of_constraints, problem_instance, lower_variable_bounds,
                upper_variable_bounds, lower_constraint_bounds,
                upper_constraint_bounds, initial_fluence, maximum_iterations,
                tolerance))

    def configure(
            self,
            number_of_variables,
            number_of_objectives,
            number_of_constraints,
            problem_instance,
            lower_variable_bounds,
            upper_variable_bounds,
            lower_constraint_bounds,
            upper_constraint_bounds,
            initial_fluence,
            maximum_iterations,
            tolerance):
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

        initial_fluence : ndarray
            Initial fluence vector.

        maximum_iterations : int
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

        # Set the number of evaluation points
        number_of_points = 200

        # Get the reference directions
        reference_directions = get_reference_directions(
            "energy", number_of_objectives, number_of_points, seed=1)

        # Initialize and evaluate the initial population
        initial_population = Population.new(
            "X", 2*max(initial_fluence)*beta(
                a=0.5, b=0.5, size=(number_of_points, number_of_variables))
            )

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
            xtol=1e-12, ftol=tolerance, n_max_gen=maximum_iterations)

        return fun, algorithm_object, problem, termination

    def run(
            self,
            _):
        """
        Run the Pymoo solver.

        Parameters
        ----------
        initial_fluence : ndarray
            Initial fluence vector.

        Returns
        -------
        ndarray
            Optimized (Pareto) set of fluence vectors.

        str
            Description for the cause of termination.
        """

        # Solve the optimization problem
        result = self.fun(
            self.problem, self.algorithm_object, self.termination, seed=1,
            save_history=False, verbose=False, callback=CustomCallback())

        return result.X, result.message


class CustomCallback(Callback):
    """
    Custom callback object for the Pymoo solver.

    Attributes
    ----------
    objective_names : tuple
        Tuple with the compound names of the objective functions.

    constraint_names : tuple
        Tuple with the compound names of the constraint functions.
    """

    def __init__(self):

        # Call the superclass constructor
        super().__init__()

        # Get the objective names
        self.objective_names = tuple(
            f"mod. {objective.name}-{segment}" for objective, segment in zip(
                get_all_objectives(Datahub().segmentation),
                get_objective_segments(Datahub().segmentation)))

        # Get the constraint names
        self.constraint_names = tuple(
            f"{constraint.name}-{segment}" for constraint, segment in zip(
                get_all_constraints(Datahub().segmentation),
                get_constraint_segments(Datahub().segmentation)))

    def notify(
            self,
            algorithm):
        """
        Log the intermediate results after each iteration.

        Parameters
        ----------
        algorithm : object of class from :mod:`pymoo.algorithms`
            The object representing the solution algorithm.
        """

        # Set the base output string
        output_string = ', '.join((
            f"{round(value, 4)} ({name})" for value, name in zip(
                mean(algorithm.pop.get("F"), axis=0), self.objective_names)))

        # Check if any constraints have been passed
        if len(self.constraint_names) > 0:

            # Get the mean values for each constraint
            values = mean(algorithm.pop.get("G"), axis=0)

            # Get the additional string
            add_string = ', '.join((
                f"{[round(values[i], 4), round(values[i+1], 4)]} ({name})"
                for i, name in enumerate(self.constraint_names)))

            # Extend the output string
            output_string = f"{output_string}, {add_string}"

        # Log a message about the intermediate mean component values
        Datahub().logger.display_info(
            f"At generation {algorithm.n_gen}: {output_string}")


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

    lower_constraint_bounds : list
        See 'Parameters'.

    upper_constraint_bounds : list
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
        super().__init__(
            n_var=number_of_variables,
            n_obj=number_of_objectives,
            n_ieq_constr=2*number_of_constraints,
            xl=array(lower_variable_bounds),
            xu=array(upper_variable_bounds))

        # Get the instance attributes from the arguments
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

        # Get the objective function values
        objective_values = self.problem_instance.objective(x)

        # Set the mixture parameter
        alpha = 0.5

        # Set the mixture fitness values
        out['F'] = [
            (1-alpha)*value + alpha*sum(objective_values)/len(objective_values)
            for value in objective_values]

        # Get the constraint function values
        constraint_values = self.problem_instance.constraint(x)

        # Set the constraint values
        out['G'] = [[
            self.lower_constraint_bounds[index]-value,
            value - self.upper_constraint_bounds[index]]
            for index, value in enumerate(constraint_values)]
