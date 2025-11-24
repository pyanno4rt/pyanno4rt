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

from pyanno4rt.logging import get_logger

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
    algorithm : str
        Label for the solution algorithm.

    maximum_iterations : int
        Maximum number of iterations.

    tolerance : float
        Precision goal for the objective function value.

    Attributes
    ----------
    algorithm : str
        See 'Parameters'.

    maximum_iterations : int
        See 'Parameters'.

    tolerance : float
        See 'Parameters'.

    fun : callable
        Minimization function from the Pymoo library.

    pymoo_algorithm : object of class from :mod:`pymoo.algorithms`
        The object used to represent the solution algorithm.

    pymoo_problem : object of class from :mod:`pymoo.core.problem`
        The object used to represent the Pymoo-compatible structure of the \
        multi-objective (Pareto) optimization problem.

    termination : object of class from :mod:`pymoo.termination`
        The object used to represent the termination criterion.
    """

    def __init__(
            self,
            algorithm,
            maximum_iterations,
            tolerance):

        # Log a message about the initialization of the class
        get_logger().info(
            "Initializing Pymoo solver with %s algorithm ...", algorithm)

        # Get the input arguments
        self.algorithm = algorithm
        self.maximum_iterations = maximum_iterations
        self.tolerance = tolerance

        # Initialize the function, algorithm, problem and termination
        self.fun, self.pymoo_alg, self.pymoo_prob, self.termination = (
            None, None, None, None)

    def configure(
            self,
            problem):
        """
        Configure the Pymoo solver.

        Supported algorithms: NSGA-3.

        Parameters
        ----------
        problem : object of class \
            :class:`~pyanno4rt.optimization.problems._pareto_problem.ParetoProblem`
            The object used to represent the optimization problem.
        """

        # Set the optimization function
        self.fun = minimize

        # Initialize the Pymoo problem instance
        self.pymoo_prob = PymooProblem(problem=problem)

        # Set the number of evaluation points
        number_of_points = 200

        # Get the reference directions
        reference_directions = get_reference_directions(
            "energy", len(problem.objectives), number_of_points, seed=1)

        # Initialize and evaluate the initial population
        initial_population = Population.new(
            "X", 2*max(problem.initial_fluence)*beta(a=0.5, b=0.5, size=(
                number_of_points, len(problem.initial_fluence))))

        # Initialize the NSGA-3 algorithm
        self.pymoo_alg = NSGA3(
            ref_dirs=reference_directions,
            pop_size=number_of_points,
            n_offsprings=number_of_points,
            sampling=initial_population,
            crossover=UniformCrossover(prob=1.0),
            mutation=PM(prob=1/len(problem.initial_fluence), eta=20),
            eliminate_duplicates=True)

        # Check if no constraints have been passed
        if len(problem.constraints) == 0:

            # Initialize the termination instance
            self.termination = DefaultMultiObjectiveTermination(
                xtol=1e-12, cvtol=1e-8, ftol=self.tolerance, n_skip=5,
                period=50, n_max_gen=self.maximum_iterations)

    def run(
            self,
            _):
        """
        Run the Pymoo solver.

        Returns
        -------
        ndarray
            Optimized (Pareto) set of fluence vectors.

        str
            Description for the cause of termination.
        """

        # Solve the optimization problem
        result = self.fun(
            self.pymoo_prob, self.pymoo_alg, self.termination, seed=1,
            save_history=False, verbose=False,
            callback=CustomCallback(self.pymoo_prob.problem))

        return result.X, result.message


class CustomCallback(Callback):
    """
    Custom callback object for the Pymoo solver.

    Parameters
    ----------
    problem : object of class \
        :class:`~pyanno4rt.optimization.problems._pareto_problem.ParetoProblem`\
        The object used to represent the optimization problem.

    Attributes
    ----------
    problem : object of class \
        :class:`~pyanno4rt.optimization.problems._pareto_problem.ParetoProblem`\
        See 'Parameters'.
    """

    def __init__(
            self,
            problem):

        # Call the superclass constructor
        super().__init__()

        # Get the optimization problem
        self.problem = problem

    def notify(
            self,
            algorithm):
        """
        Log the intermediate results after each iteration.

        Parameters
        ----------
        algorithm : object of class from :mod:`pymoo.algorithms`
            The object used to represent the solution algorithm.
        """

        # Get the mean objective values
        objectives = dict(zip(
            self.problem.objectives, mean(algorithm.pop.get("F"), axis=0)))

        # Set the base output string
        output_string = ', '.join((
            f"{round(value, 4)} ({label})"
            for label, value in objectives.items()))

        # Check if any constraints have been passed
        if len(self.problem.constraints) > 0:

            # Get the mean constraint values
            values = mean(algorithm.pop.get("G"), axis=0)

            # Generate the constraint dictionary
            constraints = dict(zip(
                self.problem.constraints,
                (self.problem.constraint_bounds[0][i] - values[2*i]
                 for i, _ in enumerate(self.problem.constraints))))

            # Get the additional string
            add_string = ', '.join((
                f"{round(value, 4)} ({label})"
                for label, value in constraints.items()))

            # Extend the output string
            output_string = f"{output_string}, {add_string}"

        # Log a message about the intermediate mean component values
        get_logger().info(
            "At generation %s: %s", algorithm.n_gen, output_string)


class PymooProblem(ElementwiseProblem):
    """
    Pymoo optimization problem class.

    This class provides a Pymoo-compatible structure of the problem and a \
    method to evaluate the fitness of the solution in each iteration.

    Parameters
    ----------
    problem : object of class \
        :class:`~pyanno4rt.optimization.problems._pareto_problem.ParetoProblem`
        The object used to represent the optimization problem.

    Attributes
    ----------
    problem : object of class \
        :class:`~pyanno4rt.optimization.problems._pareto_problem.ParetoProblem`
        See 'Parameters'.

    Notes
    -----
    Fitness evaluation is based on a modification suggested in the paper by \
    Pang et al. (2020): DOI 10.1109/ACCESS.2020.3032240.
    """

    def __init__(
            self,
            problem):

        # Call the superclass constructor
        super().__init__(
            n_var=len(problem.initial_fluence),
            n_obj=len(problem.objectives),
            n_ieq_constr=2*len(problem.constraints),
            xl=array(problem.variable_bounds[0]),
            xu=array([1e1]*len(problem.variable_bounds[1])))

        # Get the problem instance
        self.problem = problem

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
        objective_values = self.problem.objective(x)

        # Set the mixture parameter
        alpha = 0.5

        # Set the mixture fitness values
        out['F'] = [
            (1-alpha)*value + alpha*sum(objective_values)/len(objective_values)
            for value in objective_values]

        # Get the constraint function values
        constraint_values = self.problem.constraint(x)

        # Set the constraint values
        out['G'] = [[
            self.problem.constraint_bounds[0][index] - value,
            value - self.problem.constraint_bounds[1][index]]
            for index, value in enumerate(constraint_values)]
