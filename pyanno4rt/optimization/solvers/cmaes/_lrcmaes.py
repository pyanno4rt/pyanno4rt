"""Low-rank covariance matrix adaptation evolution strategy (LR-CMA-ES)."""

# Authors: Tim Ortkamp, Chinmay Patwardhan, Pia Stammer

# %% External package import

from time import time

from collections import deque
from math import inf
from numpy import (
    arange, argmin, argsort, array, clip, copy, exp, eye, full, log, maximum,
    median, sqrt, vstack, zeros)
from numpy import sum as nsum
from numpy.linalg import inv
from numpy.random import RandomState
from scipy.linalg import norm

# %% Internal package import

from pyanno4rt.optimization.solvers.cmaes import LowRankIntegrator

# %% Low-rank covariance matrix adaptation evolution algorithm


class LRCMAES:
    """
    Low-rank covariance matrix adaptation evolution strategy class.

    This class implements the LR-CMA-ES algorithm.

    Parameters
    ----------
    number_of_variables : int
        Dimension of the search space (number of decision variables).

    objective : Callable[[ndarray], float]
        The objective function to be minimized. Must accept a 1D ``ndarray`` \
        and return a scalar ``float``.

    gradient : Callable[[ndarray], ndarray], optional
        Optional gradient function used for hybrid evolution steps.

    lower_variable_bounds : ndarray, default=None
        Lower bounds on the decision variables. Must be a 1D array of length \
        `number_of_variables`. Defaults to -inf for all variables.

    upper_variable_bounds : ndarray, default=None
        Upper bounds on the decision variables. Must be a 1D array of length \
        `number_of_variables`. Defaults to +inf for all variables.

    number_of_individuals : int, default=None
        Population size. Defaults to 4 + int(3*log(`number_of_variables`)).

    initial_sigma : float, default=0.3
        Initial step size (standard deviation).

    low_rank_integrator : {'augBUG'}, default='augBUG'
        Name of the low-rank integrator.

    low_rank_dimension : int, default=1000
        Initial rank of the approximation.

    low_rank_tolerance : float, default=1e-1
        Tolerance of the rank truncation.

    maximum_iterations : int, default=1000
        Maximum number of generations (iterations) to run before stopping.

    maximum_wall_time : int or float, default=7200
        Maximum allowed wall-clock time in seconds.

    fitness_threshold : int or float, default=-inf
        Target fitness value. If the objective value reaches this threshold, \
        optimization stops (success criterion).

    fitness_window_size : int, default=20
        Number of past iterations to consider for the median fitness \
        stagnation check.

    tolerance : float, default=1e-3
        Absolute and relative termination tolerance: stops if the change in \
        median fitness over `fitness_window_size` is below this value.

    sigma_threshold : float, default=1e-3
        Minimum allowed step size. If the step size falls below this limit, \
        optimization stops (convergence/collapse criterion).

    update_interval : int, default=1
        Frequency of the SVD update (in generations). Larger values (e.g. 10) \
        can significantly speed up the algorithm for high-dimensional problems.

    callback : Callable[[dict], None], default=None
        Optional function called at the end of each iteration. Must accept a \
        dictionary with the current results.
    """

    def __init__(
            self,
            number_of_variables,
            objective,
            gradient=None,
            lower_variable_bounds=None,
            upper_variable_bounds=None,
            number_of_individuals=None,
            initial_sigma=0.3,
            low_rank_integrator='augBUG',
            low_rank_dimension=1000,
            low_rank_tolerance=1e-1,
            maximum_iterations=1000,
            maximum_wall_time=7200,
            fitness_threshold=-inf,
            fitness_window_size=20,
            tolerance=1e-3,
            sigma_threshold=1e-3,
            update_interval=1,
            callback=None):

        # Set the random seed
        self._rng = RandomState(42)

        # Initialize the optimization problem variables
        self._number_of_variables = number_of_variables
        self.objective = objective
        self.gradient = gradient
        self.lower_variable_bounds = (
            full(self._number_of_variables, -inf)
            if lower_variable_bounds is None else lower_variable_bounds)
        self.upper_variable_bounds = (
            full(self._number_of_variables, inf)
            if upper_variable_bounds is None else upper_variable_bounds)

        # Initialize the dynamical low-rank integrator
        self.integrator = LowRankIntegrator(
            name=low_rank_integrator,
            rank=low_rank_dimension,
            truncation_tolerance=low_rank_tolerance,
            N_conserved_basis=0,
            K_step=self.K_step,
            L_step=self.L_step,
            S_step=self.S_step)

        # Initialize the update interval
        self._update_interval = update_interval

        # Initialize the population and elite sizes
        self._pop_size = (
            4 + int(3*log(self._number_of_variables))
            if number_of_individuals is None else number_of_individuals)
        self._elite_size = self._pop_size // 2 + int(self.gradient is not None)

        # Initialize the weights and variance effective selection mass
        base_weights = (
            log(self._elite_size + 0.5) - log(arange(1, self._elite_size + 1))
            )
        self._weights = base_weights / nsum(base_weights)
        self._mu_eff = 1 / nsum(self._weights**2)

        # Initialize the learning rates
        self._lr_sigma = (
            (self._mu_eff + 2) / (self.integrator.rank + self._mu_eff + 3))
        self._lr_cov = 4 / (self.integrator.rank + 4)
        self._lr_rank_1 = (
            2 * min(1, self._pop_size/6) /
            ((self.integrator.rank + 1.3)**2 + self._mu_eff))
        self._lr_rank_mu = (
            2*(self._mu_eff + 1/self._mu_eff - 2) /
            ((self.integrator.rank + 2)**2 + self._mu_eff))
        self._lr_mean = 1.0

        # Initialize the damping coefficient
        self._damp_sigma = (
            1 + 2*max(0, sqrt((self._mu_eff - 1) / self.integrator.rank) - 1)
            + self._lr_sigma)

        # Initialize the expected path length
        self._expected_path_length = (
            sqrt(self._number_of_variables) * (
                1
                - 1/(4*self._number_of_variables)
                + 1/(21*self._number_of_variables**2))
            )

        # Initialize the adaptive variables
        self._wall_start = None
        self._opt_iter = 0
        self._sigma = initial_sigma
        self._path_sigma = zeros(self._number_of_variables)
        self._path_cov = zeros((self._number_of_variables, 1))
        self._mean = zeros(self._number_of_variables)
        self._left_svec = eye(self._number_of_variables, self.integrator.rank)
        self._svals = eye(self.integrator.rank)
        self._root_cov = eye(self._number_of_variables, self.integrator.rank)

        # Initialize the stopping criteria and tracking variables
        self.maximum_iterations = maximum_iterations
        self.maximum_wall_time = maximum_wall_time
        self.fitness_threshold = fitness_threshold
        self.sigma_threshold = (
            0.0 if sigma_threshold is None else sigma_threshold)
        self.tolerance = tolerance
        self._fitness_history = deque(maxlen=fitness_window_size)
        self._callback = callback
        self._result = {
            'optimal_point': None, 'optimal_value': inf, 'solver_info': None,
            'wall_time': None}

    def K_step(
            self,
            US,
            V,
            dt):
        """."""

        #
        rank_1_term = self._lr_rank_1 * self._path_cov @ (self._path_cov.T @ V)

        #
        weighted_steps_tr = self._weights[:, None] * (self.elite_steps @ V)
        rank_mu_term = self._lr_rank_mu * self.elite_steps.T @ weighted_steps_tr

        return US + rank_1_term + rank_mu_term

    def L_step(
            self,
            U,
            VS,
            dt):
        """."""

        #
        rank_1_term = self._lr_rank_1 * self._path_cov @ (self._path_cov.T @ U)

        #
        weighted_steps_tr = self._weights[:, None] * (self.elite_steps @ U)
        rank_mu_term = self._lr_rank_mu * self.elite_steps.T @ weighted_steps_tr

        return VS + rank_1_term + rank_mu_term

    def S_step(
            self,
            U,
            S,
            V,
            U1,
            S1,
            V1,
            dt):
        """."""

        #
        path_cov_proj = U.T @ self._path_cov

        #
        elite_proj = self.elite_steps @ U

        #
        rank_1_term = self._lr_rank_1 * (path_cov_proj @ path_cov_proj.T)
        rank_mu_term = self._lr_rank_mu * (
            elite_proj.T @ (self._weights[:, None] * elite_proj))

        return S + rank_1_term + rank_mu_term

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
        zsamples = self._rng.standard_normal(
            (self._pop_size, self._root_cov.shape[1]))

        # Sample steps from the multivariate Gaussian
        steps = zsamples @ self._root_cov.T

        # Sample the new population
        population = self._mean + self._sigma*steps

        # Check if a gradient has been provided
        if self.gradient is not None:

            # Compute the gradient
            gradient = self.gradient(self._mean)

            # Compute the unscaled natural gradient
            natural_gradient = (self._root_cov@self._root_cov.T) @ gradient

            # Compute the rescaling factor
            rescale = 1 / (sqrt(gradient @ natural_gradient) + 1e-15)

            # Compute the natural gradient step
            gradient_step = self._sigma * natural_gradient * rescale

            # Extend the steps (with mirroring)
            steps = vstack([steps, (-gradient_step, +gradient_step)])

            # Extend the population (with mirroring)
            population = vstack([
                population,
                (self._mean - gradient_step, self._mean + gradient_step)])

        # Get the "feasible" population
        clip(
            population, a_min=self.lower_variable_bounds,
            a_max=self.upper_variable_bounds, out=population)

        return population, steps

    def evaluate(
            self,
            population):
        """
        Evaluate the fitness of the population and track the global optimum.

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
            self.objective(individual, track=False)
            for individual in population])

        # Get the best fitness
        best_index = argmin(fitness)
        best_fitness = fitness[best_index]

        # Append the best fitness to the history
        self._fitness_history.append(best_fitness)

        # Check if an improved solution has been found
        if self._result['optimal_value'] - best_fitness > 0:

            # Update the optimal value and point
            self._result['optimal_value'] = best_fitness
            self._result['optimal_point'] = copy(population[best_index])

        # Re-evaluate the current best individual for tracking
        self.objective(self._result['optimal_point'])

        return fitness

    def tell(
            self,
            fitness,
            steps):
        """
        Update the adaptive variables.

        Parameters
        ----------
        fitness : ndarray
            Fitness values of the new population.

        steps : ndarray
            Sample steps.
        """

        # Get the indices of the elite fitness values
        elite_indices = argsort(fitness)[:self._elite_size]

        # Get the elite step vectors
        self.elite_steps = steps[elite_indices]

        # Compute the mean of the elite step vectors
        elite_mean_step = self._weights @ self.elite_steps

        # Calculate the inverse rooted singular values with epsilon correction
        inv_root_svals = inv(sqrt(self._svals) + 1e-15)

        # Transform the elite mean step
        elite_mean_step_tr = (
            self._left_svec
            @ (inv_root_svals @ (self._left_svec.T @ elite_mean_step))
            )

        # Update the step-size evolution path
        self._path_sigma *= (1 - self._lr_sigma)
        self._path_sigma += (
            sqrt(self._lr_sigma * (2-self._lr_sigma) * self._mu_eff)
            * elite_mean_step_tr)

        # Get the norm of the step-size evolution path
        ps_norm = norm(self._path_sigma)

        # Compute the update switch for the covariance matrix
        update_switch = (
            1.0
            if ps_norm / sqrt(1 - (1-self._lr_sigma)**(2*(self._opt_iter + 1)))
            < (1.4 + 2/(self.integrator.rank+1)) * self._expected_path_length
            else 0.0)

        # Update the rank-1 evolution path
        self._path_cov *= 1-self._lr_cov
        self._path_cov += (
            update_switch
            * sqrt(self._lr_cov * (2-self._lr_cov) * self._mu_eff)
            * elite_mean_step[:, None])

        # Compute the CMA-ES mean step
        self._mean += self._lr_mean * self._sigma * elite_mean_step

        # Update the step size
        self._sigma *= clip(
            exp((self._lr_sigma / self._damp_sigma)
                * (ps_norm / self._expected_path_length - 1)),
            a_min=1e-15, a_max=1)

        # Check if the SVD factors should be updated
        if self._opt_iter % self._update_interval == 0:

            # Update the SVD factors
            self._left_svec, self._svals, _ = self.integrator.update(
                self._left_svec, self._svals, self._left_svec, self._lr_cov)

            # Clip the singular values
            maximum(self._svals, 1e-12, out=self._svals)

            # Update the sampling matrix
            self._root_cov = self._left_svec @ sqrt(self._svals)

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
        if len(self._fitness_history) == self._fitness_history.maxlen:

            # Converthe history to a list
            history = list(self._fitness_history)

            # Get the first and second half median
            first_median = median(history[:len(history) // 2])
            second_median = median(history[len(history) // 2:])

            # Check if the absolute median difference is below tolerance
            if abs(first_median - second_median) < self.tolerance:

                # Add the solver info
                self._result['solver_info'] = (
                    'ABSOLUTE_FITNESS_PLATEAU_REACHED')

                return True

            # Get the relative median difference
            diff = (
                abs(first_median - second_median) /
                (abs(second_median) + 1e-15))

            # Check if the relative median difference is below tolerance
            if diff < self.tolerance:

                # Add the solver info
                self._result['solver_info'] = (
                    'RELATIVE_FITNESS_PLATEAU_REACHED')

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
