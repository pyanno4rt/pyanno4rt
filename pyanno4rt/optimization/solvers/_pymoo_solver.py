"""Pymoo wrapper."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>
# Package: https://pymoo.org/

# %% External package import

from numpy import array, clip, mean
from numpy.random import uniform

from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.population import Population
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.termination.robust import RobustTermination
from pymoo.util.display.column import Column
from pymoo.util.display.output import Output
from pymoo.util.ref_dirs import get_reference_directions

# %% Internal package import

from pyanno4rt.datahub import Datahub
from pyanno4rt.tools import get_objectives

# %% Class definition


class PymooSolver():
    """
    Pymoo wrapper class.

    This class provides methods for wrapping the Pymoo solver, including the \
    initialization of the algorithms from the arguments set in the treatment \
    plan, the composition of a Pymoo-compatible optimization problem, the \
    algorithm and termination instances, and a method to start the algorithms.

    Parameters
    ----------
    number_of_variables : int
        Number of decision variables.

    number_of_constraints : int
        Number of constraints.

    problem_instance : object of class `LexicographicOptimization`, \
    `ParetoOptimization`, or `WeightedSumOptimization`
        Instance of the optimization problem.

    lower_variable_bounds : list
        Lower bounds on the decision variables.

    upper_variable_bounds : list
        Upper bounds on the decision variables.

    lower_constraint_bounds : list
        Lower bounds on the constraints.

    upper_constraint_bounds : list
        Upper bounds on the constraints.

    algorithm : string
        Label for the solution algorithm.

    max_iter : int
        Maximum number of iterations taken for the solver to converge.

    max_cpu_time : float
        Maximum CPU time taken for the solver to converge.

    Attributes
    ----------
    problem : object of class `PymooProblem`
        Instance of the class `PymooProblem`, which represents the multi-\
        objective optimization problem.

    algorithm : object of class `NSGA3`
        Instance of the class `NSGA3`, which represents the solution algorithm.

    termination : object of class `DefaultMultiObjectiveTermination`
        Instance of the class `DefaultMultiObjectiveTermination`, which \
        represents the problem termination criteria.
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
            max_iter,
            max_cpu_time):

        # Initialize the datahub
        hub = Datahub()

        # Log a message about the initialization of the class
        hub.logger.display_info("Initializing Pymoo solver with {} ..."
                                .format(algorithm))

        # Initialize the Pymoo problem instance
        self.problem = PymooProblem(
            number_of_variables=number_of_variables,
            number_of_objectives=len(get_objectives(hub.segmentation)),
            problem_instance=problem_instance,
            lower_variable_bounds=lower_variable_bounds,
            upper_variable_bounds=[1e12]*len(upper_variable_bounds))

        # Set the number of points to evaluate
        number_of_points = 100

        # Initialize and evaluate the initial population
        initial_population = Population.new(
            "X", clip(uniform(
                low=0, high=0.5323,
                size=(number_of_points, number_of_variables)),
                a_min=0, a_max=None))

        # Get the reference directions from the Riesz s-Energy
        reference_directions = get_reference_directions(
            "energy", len(get_objectives(hub.segmentation)), number_of_points,
            seed=1)

        # Initialize the algorithm instance
        if algorithm == 'NSGA3':
            self.algorithm = NSGA3(ref_dirs=reference_directions,
                                   pop_size=number_of_points,
                                   n_offsprings=number_of_points,
                                   sampling=initial_population,
                                   crossover=SBX(prob=0.9, eta=20),
                                   mutation=PM(prob=1/number_of_variables,
                                               eta=20),
                                   eliminate_duplicates=True)

        # Initialize the termination instance
        self.termination = RobustTermination(
            DefaultMultiObjectiveTermination(
                ftol=1e-3, n_max_gen=max_iter), period=30)

    def __str__(self):
        """
        Print the class attributes.

        Returns
        -------
        string
            Class attributes as a formatted string.
        """
        return '\n'.join(("Pymoo wrapper class attributes:",
                          "----------------------------------",
                          str((*self.__dict__,))))

    def start(
            self,
            initial_fluence):
        """
        Run the Pymoo solver.

        Parameters
        ----------
        initial_fluence : ndarray
            Initial fluence vector.

        Returns
        -------
        result.X
            Optimized fluence vector.

        result.message
            Description of the cause of termination.
        """
        # Solve the optimization problem
        result = minimize(self.problem, self.algorithm, self.termination,
                          seed=1, save_history=False, verbose=True,
                          output=CustomOutput(
                              len(get_objectives(Datahub().segmentation))))

        return result.X, result.message


class PymooProblem(ElementwiseProblem):
    """
    Pymoo optimization problem class.

    Parameters
    ----------
    number_of_variables : int
        Number of decision variables.

    number_of_objectives : int
        Number of objective functions.

    problem_instance : object of class `LexicographicOptimization`, \
    `ParetoOptimization`, or `WeightedSumOptimization`
        Instance of the optimization problem.

    lower_variable_bounds : ndarray
        Lower bounds on the decision variables.

    upper_variable_bounds : ndarray
        Upper bounds on the decision variables.

    Attributes
    ----------
    problem_instance : object of class `LexicographicOptimization`, \
    `ParetoOptimization`, or `WeightedSumOptimization`
        See 'Parameters'.
    """

    def __init__(
            self,
            number_of_variables,
            number_of_objectives,
            problem_instance,
            lower_variable_bounds,
            upper_variable_bounds):

        # Call the superclass constructor
        super().__init__(n_var=number_of_variables,
                         n_obj=number_of_objectives,
                         xl=array(lower_variable_bounds),
                         xu=array(upper_variable_bounds))

        # Get the problem instance from the argument
        self.problem_instance = problem_instance

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
            Values of the decision variables.

        out : dict
            Dictionary with the objective and constraint values.

        args : tuple
            Non-keyworded (optional) input arguments.

        kwargs : tuple
            Keyworded (optional) input arguments.
        """
        # Insert the vector of objective values into the dictionary
        # value = self.problem_instance.objective(x)
        # alpha = 0.1
        # out['F'] = [(1-alpha)*val + (alpha/len(value))*sum(value)
        #             for val in value]
        out['F'] = self.problem_instance.objective(x)

        # Insert the vector of constraint values into the dictionary
        out['G'] = self.problem_instance.constraint(x)


class CustomOutput(Output):
    """."""

    def __init__(
            self,
            number_of_objectives):

        # Call the superclass constructor
        super().__init__()

        # Get the instance attribute from the argument
        self.number_of_objectives = number_of_objectives

        # Set the variables for the output columns
        for n in range(self.number_of_objectives):
            name = ''.join(('fmean_', str(n+1)))
            setattr(self, name, Column(name, width=8))
            self.columns.append(getattr(self, name))

    def update(
            self,
            algorithm):
        """."""
        # Update the algorithm
        super().update(algorithm)

        # Get the current means for all objective space dimension
        objective_mean_per_dimension = mean(algorithm.pop.get("F"), axis=0)

        # Set the values for the output columns
        for n in range(self.number_of_objectives):
            name = ''.join(('fmean_', str(n+1)))
            attr = getattr(self, name)
            attr.set(objective_mean_per_dimension[n])
