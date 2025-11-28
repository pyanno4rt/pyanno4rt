"""Covariance matrix adaptation evolution strategy (CMA-ES)."""

# Author: Tim Ortkamp

# %% External package import

from time import time

from math import inf
from numpy import (
    arange, argmin, argpartition, array, einsum, exp, eye, log, maximum,
    minimum, zeros)
from numpy import sum as nsum
from numpy.linalg import cholesky, inv, norm, svdvals
from numpy.random import seed, standard_normal

# %% Covariance matrix adaptation evolution algorithm


class CMAES:
    """
    Covariance matrix adaptation evolution strategy class.

    This class implements the CMA-ES algorithm.

    Parameters
    ----------


    Attributes
    ----------

    """

    # Set the random seed
    seed(42)

    def __init__(
            self,
            number_of_variables,
            objective,
            lower_variable_bounds,
            upper_variable_bounds,
            gradient=None,
            maximum_iterations=100,
            tolerance=1e-4,
            early_stopping_rounds=None,
            number_of_individuals=None,
            callback=None
            ):

        # Initialize the optimization problem variables
        self._number_of_variables = number_of_variables
        self.objective = objective
        self.gradient = gradient
        self.lower_bound = lower_variable_bounds
        self.upper_bound = upper_variable_bounds

        # Initialize the stopping criteria variables
        self.maximum_iterations = maximum_iterations
        self.tolerance = tolerance
        self.early_stopping_threshold = tolerance
        self.early_stopping_rounds = (
            inf if early_stopping_rounds is None else early_stopping_rounds)

        # Initialize the callback variable
        self._callback = callback

        # Initialize the fixed algorithm variables
        self._zeros = zeros(self._number_of_variables)
        self._pop_size = (
            4 + int(3*log(self._number_of_variables))
            if number_of_individuals is None else number_of_individuals)
        self._elite_size = int(self._pop_size/2)
        self._weights = (
            (log((self._elite_size+0.5)/arange(1, self._elite_size+1)))
            / nsum(log((self._elite_size+0.5)/arange(1, self._elite_size+1))))
        self._mu_eff = 1/nsum(self._weights)
        self._lr_sigma = (
            (self._mu_eff+2) / (self._number_of_variables+self._mu_eff+3))
        self._lr_cov = 4/(self._number_of_variables+4)
        self._lr_rank_1 = (
            (2*min(1, self._pop_size/6))
            / ((self._number_of_variables+1.3)**2+self._mu_eff))
        self._lr_rank_mu = (
            2*(self._mu_eff+1/self._mu_eff-2)
            / ((self._number_of_variables+2)**2+self._mu_eff))
        self._lr_mean = 1
        self._damp_sigma = (
            1+2*max(0, ((self._mu_eff-1)/self._number_of_variables)**0.5 - 1)
            + self._lr_sigma)
        self._expected_path_length = (
            self._number_of_variables**0.5
            * (1-1/(4*self._number_of_variables)
               + 1/(21*self._number_of_variables**2)))

        # Initialize the adaptive algorithm variables
        self._opt_iter = 0
        self._early_stopping_iter = 0
        self._sigma = 0.3
        self._path_sigma = zeros(self._number_of_variables)
        self._path_cov = zeros(self._number_of_variables)
        self._mean = zeros(self._number_of_variables)
        self._cov = eye(self._number_of_variables)

        # Initialize the singular values list
        self._singular_values = []

        # Initialize the result dictionary
        self._result = {
            'optimal_point': None, 'optimal_value': inf, 'solver_info': None,
            'runtime': None}

    def check_termination(self):
        """
        Check the termination criteria.

        Returns
        -------
        bool
            Indicator for termination.
        """

        # Check if the maximum number of iterations has been reached
        if self._opt_iter >= self.maximum_iterations:

            # Add the solver info
            self._result['solver_info'] = 'MAX_ITER_REACHED'

            return True

        # Check if maximum number of early stopping rounds has been reached
        if self._early_stopping_iter >= self.early_stopping_rounds:

            # Add the solver info
            self._result['solver_info'] = 'EARLY_STOPPING'

            return True

        return False

    def evaluate_fitness(
            self,
            population,
            fitness):
        """
        Evaluate the fitness result of the population.

        Parameters
        ----------
        population : ndarray
            Sample population.

        fitness : ndarray
            Fitness values for the population.

        Returns
        -------
        bool
            Indicator for termination.
        """

        # Get the best fitness
        best_fitness = fitness.min()

        # Get the fitness improvement
        fitness_improvement = self._result['optimal_value'] - best_fitness

        # Check if an improved solution has been found
        if fitness_improvement > 0:

            # Update the optimal value and point
            self._result['optimal_value'] = best_fitness
            self._result['optimal_point'] = population[argmin(fitness)]

            # Check if the improvement exceeds the early stopping threshold
            if fitness_improvement >= self.early_stopping_threshold:

                # Reset the early stopping counter
                self._early_stopping_iter = 0

        # Check if the fitness improvement only is not sufficient
        elif (self.early_stopping_rounds == inf
                and fitness_improvement < self.tolerance):

            # Add the solver info
            self._result['solver_info'] = 'BELOW_THRESHOLD'

            return False

        else:

            # Increment the early stopping counter
            self._early_stopping_iter += 1

        return True

    def ask(self):
        """
        Generate a new population.

        Returns
        -------
        ndarray
            Sample population (bound to the feasible region).

        ndarray
            Sample steps.
        """

        # Sample from the standard multivariate Gaussian
        zsamples = standard_normal((self._pop_size, self._number_of_variables))

        # Sample steps from the multivariate Gaussian
        steps = zsamples@cholesky(self._cov)

        # Sample the new population
        population = self._mean + self._lr_mean*self._sigma*steps

        return (
            minimum(maximum(population, self.lower_bound), self.upper_bound),
            steps)

    def tell(
            self,
            fitness,
            steps):
        """
        Update the adaptive algorithm variables.

        Parameters
        ----------
        fitness : ndarray
            Fitness values of the new population.

        steps : ndarray
            Sample steps.
        """

        # Get the indices of the elite fitness values
        elite_indices = argpartition(
            fitness, self._elite_size)[:self._elite_size]

        # Get the elite step vectors
        elite_steps = steps[elite_indices, :]

        # Compute the mean of the elite step vectors
        elite_mean_step = self._weights@elite_steps

        # Update the step-size evolution path
        path_sigma = (
            (1-self._lr_sigma)*self._path_sigma
            + (self._lr_sigma*(2-self._lr_sigma)*self._mu_eff)**0.5
            * inv(cholesky(self._cov)).T
            * elite_mean_step)

        # Compute the update switch for the covariance matrix
        update_switch = (
            norm(path_sigma)
            < (1-(1-self._lr_sigma)**(2*(self._opt_iter+1)))**0.5
            *(1.4+2/(self._number_of_variables+1))*self._expected_path_length)

        # Update the rank-1 evolution path
        path_cov = (
            (1-self._lr_cov)*self._path_cov
            + update_switch
            * (self._lr_cov*(2-self._lr_cov)*self._mu_eff)**0.5
            * elite_mean_step)

        # Update the mean vector
        self._mean = self._mean + self._sigma*elite_mean_step

        # Update the step size
        self._sigma = (
            self._sigma*exp(
                (self._lr_sigma/self._damp_sigma)
                 * ((norm(path_sigma)/self._expected_path_length)-1)))

        # Update the covariance matrix
        self._cov = (
            (1+(1-update_switch)*self._lr_rank_1*self._lr_cov*(2-self._lr_cov))
            * self._cov
            + self._lr_rank_1*(path_cov@path_cov.T-self._cov)
            + self._lr_rank_mu*(
                einsum('a,ai,aj->ij', self._weights, elite_steps, elite_steps,
                       optimize=True)
                - nsum(self._weights)*self._cov))

    def optimize(
            self,
            initial_mean=None):
        """
        Run the optimization algorithm.

        Parameters
        ----------
        initial_mean : ndarray, default=None
            Initial mean vector. Default corresponds to the zero vector.

        Returns
        -------
        dict
            Dictionary with the optimization results.
        """

        # Start the runtime recording
        start = time()

        # Select the mean value
        self._mean = self._mean if initial_mean is None else initial_mean

        # Continue until termination criteria are fulfilled
        while self.check_termination() is False:

            # ---------------------------------------------- REMOVE LATER
            # Store the eigenvalues of the covariance matrix
            # self._singular_values.append(svdvals(self._cov))
            # ----------------------------------------------

            # "Ask" for a new population
            population, steps = self.ask()

            # Compute the fitness values
            fitness = array([
                self.objective(individual) for individual in population])

            # Evaluate the population's fitness
            fitness_check = self.evaluate_fitness(population, fitness)

            # Check if the fitness evaluation turned out negative
            if fitness_check is False:

                # Break the loop
                break

            # Check if a callback has been provided
            if self._callback is not None:

                # Pass the current results to the callback
                self._callback(self._result)

            # "Tell" the algorithm to update its parameters
            self.tell(fitness, steps)

            # Increment the iteration counter
            self._opt_iter += 1

        # Store the runtime
        self._result['runtime'] = time()-start

        return self._result


# %% Toy problem class

# from math import pi
# from numpy import cos, sin

# class ToyProblem:
#     """
#     A toy problem class to test the CMA algorithm with different benchmark \
#     functions.

#     Parameters
#     ----------
#     name : {'rastrigin', 'sphere', 'styblinski-tang'}
#         Name of the test function.

#     initial_x : ndarray
#         Initial decision variables.

#     Attributes
#     ----------
#     name : {'rastrigin', 'sphere', 'styblinski-tang'}
#         See 'Parameters'.

#     f : function
#         Objective function.

#     g : function
#         Gradient function.

#     variable_bounds : tuple
#         Tuple with two lists for the lower and upper variable bounds.
#     """

#     def __init__(
#             self,
#             name,
#             x0):

#         # Get the function name
#         self.name = name
#         self.x0 = x0

#         # Map the names
#         functions = {
#             'rastrigin': self.rastrigin,
#             'sphere': self.sphere,
#             'styblinski-tang': self.styblinski_tang}

#         # Get the objective and gradient functions
#         self.f, self.g, self.variable_bounds = functions[name]()

#     def objective(self, x):
#         """
#         Compute the objective.

#         Parameters
#         ----------
#         x : ndarray
#             Decision vector.

#         Returns
#         -------
#         int or float
#             Objective value.
#         """

#         return self.f(x)

#     def gradient(
#             self,
#             x):
#         """
#         Compute the gradient.

#         Parameters
#         ----------
#         x : ndarray
#             Decision vector.

#         Returns
#         -------
#         ndarray
#             Gradient.
#         """

#         return self.g(x)

#     def rastrigin(self):
#         """
#         Rastrigin function.

#         Global minimum: f(0,...,0) = 0.0, search domain: [-5.12, 5.12].
#         """

#         def f(x):
#             return 10*len(x) + sum(x**2 - 10*cos(2*pi*x))

#         def g(x):
#             return 2*x+20*pi*sin(2*pi*x)

#         bounds = ([-5.12]*len(self.x0), [5.12]*len(self.x0))

#         return f, g, bounds

#     def sphere(self):
#         """
#         Sphere function.

#         Global minimum: f(0,...,0) = 0.0, search domain: [-inf, inf].
#         """

#         def f(x):
#             return sum(x**2)

#         def g(x):
#             return 2*x

#         bounds = ([-inf]*len(self.x0), [inf]*len(self.x0))

#         return f, g, bounds

#     def styblinski_tang(self):
#         """
#         Styblinski-Tang function.

#         Global minimum: -39.16617*len(x) < f(-2.903534,...,-2.903534) \
#         < -39.16616*len(x), search domain: [-5, 5].
#         """

#         def f(x):
#             return sum(x**4-16*x**2+5*x)/2

#         def g(x):
#             return 2*x**3-16*x+2.5

#         bounds = ([-5]*len(self.x0), [5]*len(self.x0))

#         return f, g, bounds


# %% Test run

# # Set the initial vector
# initial_x = array([0]*5)

# # Initialize the toy problem
# prob = ToyProblem(
#     name='rastrigin',
#     x0=initial_x)

# # Initialize the CMA solver
# solver = CMAES(
#     number_of_variables=len(initial_x),
#     objective=prob.f,
#     lower_variable_bounds=array(prob.variable_bounds[0]),
#     upper_variable_bounds=array(prob.variable_bounds[1]),
#     gradient=prob.g,
#     maximum_iterations=10000,
#     tolerance=1e-3,
#     early_stopping_rounds=100)

# # Optimize the variables
# result = solver.optimize(initial_x)

# # Get the iteration-wise eigenvalues
# sv = solver._singular_values
