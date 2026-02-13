"""Covariance matrix adaptation evolution strategy (CMA-ES)."""

# Author: Tim Ortkamp

# %% External package import

from time import time

from collections import deque
from math import inf
from numpy import (
    arange, argmin, argpartition, array, diag, exp, eye, hstack, log, maximum,
    median, minimum, sqrt, zeros)
from numpy import sum as nsum
from numpy.linalg import inv, norm, qr, svd
from numpy.random import seed, standard_normal

# %% Covariance matrix adaptation evolution algorithm


class LRCMAES:
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
            lower_variable_bounds=None,
            upper_variable_bounds=None,
            gradient=None,
            number_of_individuals=None,
            initial_sigma=None,
            maximum_iterations=1000,
            maximum_wall_time=7200,
            fitness_threshold=None,
            fitness_window_size=20,
            sigma_threshold=1e-3,
            tolerance=1e-3,
            callback=None):

        # Initialize the optimization problem variables
        self._number_of_variables = number_of_variables
        self.objective = objective
        self.gradient = gradient
        self.lower_bound = (
            [-inf]*number_of_variables if lower_variable_bounds is None
            else lower_variable_bounds)
        self.upper_bound = (
            [-inf]*number_of_variables if upper_variable_bounds is None
            else upper_variable_bounds)

        # Initialize the stopping criteria variables
        self.maximum_iterations = maximum_iterations
        self.maximum_wall_time = maximum_wall_time
        self.fitness_threshold = (
            -inf if fitness_threshold is None else fitness_threshold)
        self.sigma_threshold = (
            0 if sigma_threshold is None else sigma_threshold)
        self.fitness_window_size = fitness_window_size
        self.tolerance = tolerance

        # Initialize the callback variable
        self._callback = callback

        # Initialize the fitness history
        self._fitness_history = None

        # Initialize the singular values list
        self._singular_values = []

        # Initialize the fixed algorithm variables
        self._zeros = zeros(self._number_of_variables)
        self._pop_size = (
            4 + int(3*log(self._number_of_variables))
            if number_of_individuals is None else number_of_individuals)
        self._elite_size = int(self._pop_size/2)
        self._weights = (
            (log((self._elite_size+0.5)/arange(1, self._elite_size+1)))
            / nsum(log((self._elite_size+0.5)/arange(1, self._elite_size+1))))
        self._mu_eff = 1/nsum(self._weights**2)
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
        self._wall_start = None
        self._opt_iter = 0
        self._sigma = 0.3 if initial_sigma is None else initial_sigma
        self._path_sigma = zeros(self._number_of_variables).reshape(-1, 1)
        self._path_cov = zeros(self._number_of_variables).reshape(-1, 1)
        self._mean = zeros(self._number_of_variables)

        # Perform SVD on the initial covariance matrix
        self.left_svec, self.svals, _ = svd(eye(self._number_of_variables))
        self.svals = diag(self.svals)

        # Truncate to rank r
        r = 1000
        self.left_svec = self.left_svec[:, :r]
        self.svals = self.svals[:r, :r]

        # Initialize the result dictionary
        self._result = {
            'optimal_point': None, 'optimal_value': inf, 'solver_info': None,
            'wall_time': None}

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
        steps = (self.left_svec @ self.svals @ zsamples.T[
            :self.left_svec.shape[1], :]).T

        # Sample the new population
        population = self._mean + self._lr_mean*self._sigma*steps

        return (
            minimum(maximum(population, self.lower_bound), self.upper_bound),
            steps)

    def evaluate(
            self,
            population):
        """
        Evaluate the fitness of the population.

        Parameters
        ----------
        population : ndarray
            Sample population.

        Returns
        -------
        ndarray
            Fitness values for the population.
        """

        # Compute the fitness values
        fitness = array([
            self.objective(individual) for individual in population])

        # Get the best fitness
        best_fitness = fitness.min()

        # Append the best fitness to the history
        self._fitness_history.append(best_fitness)

        # Get the fitness improvement
        fitness_improvement = self._result['optimal_value'] - best_fitness

        # Check if an improved solution has been found
        if fitness_improvement > 0:

            # Update the optimal value and point
            self._result['optimal_value'] = best_fitness
            self._result['optimal_point'] = population[argmin(fitness)]

        return fitness

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

        # Update the mean vector
        self._mean = self._mean + self._sigma*elite_mean_step

        # Update the step-size evolution path
        self._path_sigma = (
            (1-self._lr_sigma)*self._path_sigma
            + (self._lr_sigma*(2-self._lr_sigma)*self._mu_eff)**0.5
            * elite_mean_step)

        # Compute the update switch for the covariance matrix
        update_switch = (
            norm(self._path_sigma)
            < (1-(1-self._lr_sigma)**(2*(self._opt_iter+1)))**0.5
            * (1.4+2/(self._number_of_variables+1))*self._expected_path_length)

        # Update the rank-1 evolution path
        self._path_cov = (
            (1-self._lr_cov)*self._path_cov
            + update_switch
            * (self._lr_cov*(2-self._lr_cov)*self._mu_eff)**0.5
            * elite_mean_step)

        # Update the step size
        self._sigma = (
            self._sigma*exp(
                (self._lr_sigma/self._damp_sigma)
                 * ((norm(self._path_sigma)/self._expected_path_length)-1)))

        # Update the covariance matrix
        k0 = self.left_svec @ self.svals
        k1 = k0 + self._lr_cov*(
            self._path_cov@self._path_cov.T@self.left_svec - k0)
        left_svec_upd, _ = qr(k1)
        st0 = left_svec_upd.T @ self.left_svec @ self.svals @ self.left_svec.T @ left_svec_upd
        self.svals = st0 + self._lr_cov*(
            left_svec_upd.T @ self._path_cov @ self._path_cov.T @ left_svec_upd
            - st0)
        self.left_svec = left_svec_upd
        self.svals = st0

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

        # Start the runtime recordings
        self._wall_start = time()

        # Select the mean value
        self._mean = self._mean if initial_mean is None else initial_mean

        # Initialize the fitness history
        self._fitness_history = deque(maxlen=self.fitness_window_size)

        # Continue until termination criteria are fulfilled
        while self.check_termination() is False:

            # "Ask" for a new population
            population, steps = self.ask()

            # Evaluate the population's fitness
            fitness = self.evaluate(population)

            # "Tell" the algorithm to update its parameters
            self.tell(fitness, steps)

            # Check if a callback has been provided
            if self._callback is not None:

                # Pass the current results to the callback
                self._callback(self._result)

            # Increment the iteration counter
            self._opt_iter += 1

        # Store the runtimes
        self._result['wall_time'] = time()-self._wall_start

        return self._result

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

        # Check if the maximum runtime has been reached
        if time()-self._wall_start >= self.maximum_wall_time:

            # Add the solver info
            self._result['solver_info'] = 'MAX_WALL_TIME_REACHED'

            return True

        # Check if the history is completely filled
        if len(self._fitness_history) == self.fitness_window_size:

            # Converthe history to a list
            fitness_history = list(self._fitness_history)

            # Check if the split median difference is below tolerance
            if (median(fitness_history[:self.fitness_window_size//2])
                - median(fitness_history[self.fitness_window_size//2:])
                <= self.tolerance):

                # Add the solver info
                self._result['solver_info'] = 'FITNESS_PLATEAU_REACHED'

                return True

        # Check if the optimal value is below a threshold
        if self._result['optimal_value'] <= self.fitness_threshold:

            # Add the solver info
            self._result['solver_info'] = 'FITNESS_BELOW_THRESH'

            return True

        # Check if the step size is below the threshold
        if self._sigma <= self.sigma_threshold:

            # Add the solver info
            self._result['solver_info'] = 'SIGMA_BELOW_THRESH'

            return True

        return False
