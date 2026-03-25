"""Low-rank covariance matrix adaptation evolution strategy (LR-CMA-ES)."""

# Authors: Tim Ortkamp, Chinmay Patwardhan, Pia Stammer

# %% External package import

from time import time

from collections import deque
from math import inf
from numba import njit
from numpy import (
    add, arange, argmin, argsort, array, clip, copyto, empty, exp, eye,
    fill_diagonal, float64, full, log, matmul, maximum, median, multiply, ones,
    sqrt, take, zeros)
from numpy import sum as nsum
from numpy.random import default_rng
from scipy.linalg import eigh

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

    low_rank_integrator : {'fixedBUG', 'fixedsymmetricBUG', 'fixedaugBUG', \
                           'fixedSPDBUG', 'augBUG', 'symmetricaugBUG'}, \
        default='fixedsymmetricBUG'
        Name of the low-rank integrator.

    low_rank_dimension : int, default=None
        Initial rank of the approximation. Defaults to \
        `number_of_variables` // 10.

    low_rank_tolerance_rel : float, default=1e-2
        Relative tolerance of the rank truncation.

    low_rank_tolerance_abs : float, default=1e-8
        Absolute tolerance of the rank truncation.

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
        Frequency of the low-rank updates (in generations). Larger values \
        (e.g. 10) can significantly speed up the algorithm for \
        high-dimensional problems.

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
            low_rank_integrator='fixedsymmetricBUG',
            low_rank_dimension=None,
            low_rank_tolerance_rel=1e-2,
            low_rank_tolerance_abs=1e-8,
            maximum_iterations=1000,
            maximum_wall_time=7200,
            fitness_threshold=-inf,
            fitness_window_size=20,
            tolerance=1e-3,
            sigma_threshold=1e-3,
            update_interval=1,
            callback=None):

        # Set the random seed
        self._rng = default_rng(42)

        # Initialize the optimization problem variables
        self._number_of_variables = number_of_variables
        self.objective = objective
        self.gradient = gradient
        self.lower_variable_bounds = (
            full(number_of_variables, -inf)
            if lower_variable_bounds is None else lower_variable_bounds)
        self.upper_variable_bounds = (
            full(number_of_variables, inf)
            if upper_variable_bounds is None else upper_variable_bounds)

        # Determine the integrator rank
        rank = min(
            low_rank_dimension or number_of_variables // 10,
            number_of_variables)

        # Initialize the dynamical low-rank integrator
        self.integrator = LowRankIntegrator(
            name=low_rank_integrator,
            rank=rank,
            truncation_tolerance_rel=low_rank_tolerance_rel,
            truncation_tolerance_abs=low_rank_tolerance_abs,
            N_conserved_basis=0,
            K_step=self.K_step,
            L_step=self.L_step,
            S_step=self.S_step)

        # Initialize the integrator buffers
        self.integrator.set_buffers(number_of_variables)

        # Initialize the update interval
        self._update_interval = update_interval

        # Initialize the population and elite sizes
        self._pop_size = (
            4 + int(3*log(number_of_variables))
            if number_of_individuals is None else number_of_individuals)
        self._pop_size += (2 if gradient is not None else 0)
        self._elite_size = self._pop_size // 2

        # Initialize the weights and variance effective selection mass
        base_weights = (
            log(self._elite_size + 0.5) - log(arange(1, self._elite_size + 1))
            )
        self._weights = base_weights / nsum(base_weights)
        self._weights_2d = self._weights[:, None]
        self._mu_eff = 1 / nsum(self._weights**2)

        # Initialize the learning rates, damping, and expected path length
        self._update_dynamics()

        # Initialize the adaptive variables
        self._wall_start = None
        self._opt_iter = 0

        self._sigma = initial_sigma

        self._path_sigma = zeros(rank, dtype=float64)
        self._path_cov = zeros((rank, 1), order='F', dtype=float64)
        self._mean = zeros(number_of_variables, dtype=float64)

        self._left_basis = zeros(
            (number_of_variables, number_of_variables), order='F',
            dtype=float64)
        self._left_basis[:rank, :rank] = eye(rank)
        self._core_matrix = eye(rank, dtype=float64)

        self._core_eig_cache = (ones(rank), eye(rank))
        self._root_cov = zeros(
            (number_of_variables, number_of_variables), order='F',
            dtype=float64)
        self._root_cov[:rank, :rank] = eye(rank)

        # Initialize the buffer variables
        self._steps = zeros(
            (self._pop_size, number_of_variables), order='F', dtype=float64)
        self._elite_steps = zeros(
            (self._elite_size, number_of_variables), order='F', dtype=float64)
        self._population = zeros(
            (self._pop_size, number_of_variables), order='F', dtype=float64)
        self._left_basis_old = empty(
            (number_of_variables, number_of_variables), order='F',
            dtype=float64)
        self._path_cov_padded = zeros(
            (number_of_variables, 1), order='F', dtype=float64)
        self._elite_projection = zeros(
            (self._elite_size, number_of_variables), order='F', dtype=float64)
        self._path_full = empty(
            (number_of_variables, 1), order='F', dtype=float64)
        self._weighted_steps = empty(
            (self._elite_size, number_of_variables), order='F', dtype=float64)
        self._mu_projection = empty(
            (self._elite_size, number_of_variables), order='F', dtype=float64)

        # Initialize the stopping criteria and tracking variables
        self.maximum_iterations = maximum_iterations
        self.maximum_wall_time = maximum_wall_time
        self.fitness_threshold = fitness_threshold
        self.sigma_threshold = sigma_threshold or 0.0
        self.tolerance = tolerance
        self._fitness_history = deque(maxlen=fitness_window_size)
        self._callback = callback
        self._result = {
            'optimal_point': None, 'optimal_value': inf, 'solver_info': None,
            'wall_time': None}

    def K_step(
            self,
            left_basis_product,
            right_basis,
            _time_step):
        """
        Perform the K-step of the dynamical low-rank integrator.

        Parameters
        ----------
        left_basis_product :
            ...

        right_basis :
            ...

        _time_step :
            ...

        Returns
        -------
        ...
        """

        return self._update_rank_terms(left_basis_product, right_basis)

    def L_step(
            self,
            left_basis,
            right_basis_product,
            _time_step):
        """
        Perform the L-step of the dynamical low-rank integrator.

        Parameters
        ----------
        left_basis :
            ...

        right_basis_product :
            ...

        _time_step :
            ...

        Returns
        -------
        ...
        """

        return self._update_rank_terms(right_basis_product, left_basis)

    def S_step(
            self,
            left_basis,
            core_matrix,
            *args):
        """
        Perform the S-step of the dynamical low-rank integrator.

        Parameters
        ----------
        left_basis :
            ...

        core_matrix :
            ...

        *args : tuple
            ...

        Returns
        -------
        ...
        """

        # Get the ranks of the covariance evolution path and the core matrix
        path_rank = self._path_cov.shape[0]
        core_rank = core_matrix.shape[0]

        # Check if the ranks are different
        if path_rank != core_rank:

            # Add a padding to the covariance evolution path
            path_cov_padded = self._path_cov_padded[:core_rank, :]
            path_cov_padded.fill(0.0)
            path_cov_padded[:path_rank, :] = self._path_cov

        else:

            # Use the covariance evolution path directly
            path_cov_padded = self._path_cov

        # Get the projections of the covariance evolution path and elite steps
        matmul(
            self._elite_steps, left_basis,
            out=self._elite_projection[:, :core_rank])
        elite_projection = self._elite_projection[:, :core_rank]

        # Add the rank-1 update
        core_matrix += self._lr_rank_1 * (path_cov_padded @ path_cov_padded.T)

        # Add the rank-mu update
        core_matrix += self._lr_rank_mu * (
            elite_projection.T @ (self._weights_2d * elite_projection))

        return core_matrix

    def _update_rank_terms(
            self,
            target_matrix,
            projection_matrix):
        """
        Compute the rank-1 and rank-mu update terms.

        Parameters
        ----------
        target_matrix :
            ...

        projection_matrix :
            ...

        Returns
        -------
        ...
        """

        # Get the current rank
        rank = target_matrix.shape[1]

        # Compute the full-rank covariance evolution path
        rank_cma = self._path_cov.shape[0]
        matmul(
            self._left_basis[:, :rank_cma], self._path_cov, out=self._path_full
            )

        # Add the rank-1 update
        target_matrix += self._lr_rank_1 * (self._path_full @ (
            self._path_full.T @ projection_matrix))

        # Compute the weighted elite steps
        multiply(self._weights_2d, self._elite_steps, out=self._weighted_steps)

        # Project the weighted steps
        matmul(
            self._weighted_steps, projection_matrix,
            out=self._mu_projection[:, :rank])

        # Add the rank-mu update
        target_matrix += self._lr_rank_mu * (
            self._elite_steps.T @ self._mu_projection[:, :rank])

        return target_matrix

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

        # Get the current rank
        rank = self.integrator.rank

        # Sample from the standard multivariate Gaussian
        num_random = self._pop_size - (2 if self.gradient is not None else 0)
        zsamples = self._rng.standard_normal((num_random, rank))

        # Sample steps from the multivariate Gaussian
        self._steps[:num_random] = zsamples @ self._root_cov[:, :rank].T

        # Check if a gradient has been provided
        if self.gradient is not None:

            # Compute the gradient
            gradient = self.gradient(self._mean)

            # Compute the unscaled natural gradient
            natural_gradient = self._root_cov[:, :rank] @ (
                self._root_cov[:, :rank].T @ gradient)

            # Calculate the "energy" of the gradient in the subspace
            gradient_energy = gradient @ natural_gradient

            # Check if the gradient is non-orthogonal to the subspace
            if gradient_energy > 1e-12:

                # Compute the rescaling factor
                rescale = 1.0 / (sqrt(gradient_energy) + 1e-15)

                # Compute the natural gradient step
                gradient_step = natural_gradient * rescale

                # Add the gradient steps (with mirroring)
                self._steps[-2] = -gradient_step
                self._steps[-1] = gradient_step

            else:

                # Fill the remaining samples randomly
                self._steps[-2:] = (
                    self._rng.standard_normal((2, rank))
                    @ self._root_cov[:, :rank].T)

        # Sample the new population
        add(self._mean, self._sigma * self._steps, out=self._population)

        # Get the "feasible" population
        clip(
            self._population, a_min=self.lower_variable_bounds,
            a_max=self.upper_variable_bounds, out=self._population)

    def evaluate(self):
        """
        Evaluate the fitness of the population and track the global optimum.

        Returns
        -------
        ndarray
            Fitness values for the population.
        """

        # Compute the fitness values
        fitness = array([
            self.objective(individual, track=False)
            for individual in self._population])

        # Get the best fitness
        best_index = argmin(fitness)
        best_fitness = fitness[best_index]

        # Append the best fitness to the history
        self._fitness_history.append(best_fitness)

        # Check if an improved solution has been found
        if self._result['optimal_value'] - best_fitness > 0:

            # Update the optimal value and point
            self._result['optimal_value'] = best_fitness
            self._result['optimal_point'] = self._population[best_index].copy()

        # Re-evaluate the current best individual for tracking
        self.objective(self._result['optimal_point'])

        return fitness

    def tell(
            self,
            fitness):
        """
        Update the adaptive variables.

        Parameters
        ----------
        fitness : ndarray
            Fitness values of the new population.
        """

        # Get the current rank
        rank_old = self.integrator.rank

        # Get the eigenvalues and eigenvectors of the singular value matrix
        ev, eq = self._core_eig_cache

        # Update the basic variables
        self._sigma, elite_indices = _tell(
            fitness,
            self._steps,
            self._weights,
            self._left_basis[:, :rank_old],
            self._core_matrix[:rank_old, :rank_old],
            self._path_sigma,
            self._path_cov,
            self._mean,
            self._sigma,
            ev[:rank_old],
            eq[:rank_old, :rank_old],
            self._lr_sigma,
            self._lr_cov,
            self._lr_mean,
            self._mu_eff,
            self._damp_sigma,
            self._expected_path_length,
            self._opt_iter,
            self._elite_size)

        # Get the elite steps
        take(self._steps, elite_indices, axis=0, out=self._elite_steps)

        # Check if the dynamical low-rank factors should be updated
        if self._opt_iter % self._update_interval == 0:

            # Copy the current basis to the buffer
            copyto(
                self._left_basis_old[:, :rank_old],
                self._left_basis[:, :rank_old])

            # Update the dynamical low-rank factors
            U_new, S_new, _ = self.integrator.update(
                self._left_basis[:, :rank_old], self._core_matrix,
                self._left_basis[:, :rank_old], self._lr_cov)

            #
            rank_new = U_new.shape[1]

            #
            copyto(self._left_basis[:, :rank_new], U_new)
            self._core_matrix = S_new

            # Rotate the evolution paths
            projection = (
                self._left_basis[:, :rank_new].T
                @ self._left_basis_old[:, :rank_old])
            self._path_sigma = (projection @ self._path_sigma).flatten()
            self._path_cov = (projection @ self._path_cov).reshape(-1, 1)

            # Check if the rank has changed
            if rank_new != rank_old:

                # Update the dynamic parameters
                self._update_dynamics()

            # Update the eigenvalues and -vectors of the singular value matrix
            ev, eq = eigh(
                self._core_matrix, overwrite_a=True, check_finite=False)

            # Sort in descending order
            ev, eq = ev[::-1], eq[:, ::-1]

            # Clip the eigenvalues
            maximum(ev, 1e-15, out=ev)

            # Update the eigendecomposition cache
            self._core_eig_cache = (ev, eq)

            # Update the sampling matrix
            matmul(
                self._left_basis[:, :rank_new], eq * sqrt(ev),
                out=self._root_cov[:, :rank_new])

            #
            diagonal_block = self._root_cov[:, rank_new:]
            diagonal_block.fill(0.0)
            fill_diagonal(diagonal_block, sqrt(self.integrator._psi))

    def _update_dynamics(self):
        """Update the dynamic parameters for the current rank."""

        # Get the current rank
        rank = self.integrator.rank

        # Update the learning rates
        self._lr_sigma = (self._mu_eff + 2) / (rank + self._mu_eff + 3)
        self._lr_cov = 4 / (rank + 4)
        self._lr_rank_1 = 100*(
            2 * min(1, self._pop_size / 6) / ((rank + 1.3)**2 + self._mu_eff))
        self._lr_rank_mu = (
            2 * (self._mu_eff + 1/self._mu_eff - 2) /
            ((rank + 2)**2 + self._mu_eff))
        self._lr_mean = 1.0

        # Update the damping coefficient
        self._damp_sigma = (
            1
            + 2*max(0, sqrt((self._mu_eff - 1) / (rank + 1)) - 1)
            + self._lr_sigma)

        # Initialize the expected path length
        self._expected_path_length = (
            sqrt(rank) * (1 - 1/(4*rank) + 1/(21*rank**2)))

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

        # Check if an initial mean has been provided
        if initial_mean is not None:

            # Set the initial mean
            self._mean = initial_mean

        # Continue until termination criteria are fulfilled
        while self.check_termination() is False:

            # "Ask" for a new population
            self.ask()

            # Evaluate the population's fitness
            fitness = self.evaluate()

            # "Tell" the algorithm to update its parameters
            self.tell(fitness)

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

        # Check if the wall clock timer has been started
        if self._wall_start is not None:

            # Check if the maximum runtime has been reached
            if time()-self._wall_start >= self.maximum_wall_time:

                # Add the solver info
                self._result['solver_info'] = 'MAX_WALL_TIME_REACHED'

                return True

        # Check if the step size is below the threshold
        if self._sigma <= self.sigma_threshold:

            # Add the solver info
            self._result['solver_info'] = 'SIGMA_BELOW_THRESH'

            return True

        # Check if the fitness history is non-empty
        if len(self._fitness_history) > 0:

            # Check if the optimal value is below a threshold
            if self._fitness_history[-1] < self.fitness_threshold:

                # Add the solver info
                self._result['solver_info'] = 'FITNESS_BELOW_THRESH'

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

        return False


@njit(fastmath=True)
def _tell(
    fitness, steps, weights, left_basis, core_matrix, path_sigma, path_cov,
    mean, sigma, ev, eq, lr_sigma, lr_cov, lr_mean, mu_eff, damp_sigma,
    expected_path_length, opt_iter, elite_size):
    """Update the basic CMAES variables."""

    # Get the indices of the elite fitness values
    elite_indices = argsort(fitness)[:elite_size]

    # Initialize the elite mean step
    elite_mean_step = weights @ steps[elite_indices]

    # Calculate the inverse rooted eigenvalues
    inv_root_ev = 1.0 / (sqrt(ev) + 1e-15)

    # Transform the elite mean step
    latent_mean_step = left_basis.T @ elite_mean_step
    eq_step = eq.T @ latent_mean_step
    latent_step_whitened = eq @ (inv_root_ev * eq_step)

    # Update the step-size evolution path
    path_sigma *= (1.0 - lr_sigma)
    path_sigma += (
        sqrt(lr_sigma * (2.0 - lr_sigma) * mu_eff)
        * latent_step_whitened)

    # Get the norm of the step-size evolution path
    ps_norm = sqrt(nsum(path_sigma**2))

    # Compute the update switch for the covariance matrix
    ps_exp = sqrt(1.0 - (1.0 - lr_sigma)**(2.0 * (opt_iter + 1)))
    condition = (1.4 + 2.0/(core_matrix.shape[0] + 1.0)) * expected_path_length
    update_switch = 1.0 if (ps_norm / ps_exp) < condition else 0.0

    # Compute the 'keep' term of the evolution path
    path_cov *= (1.0 - lr_cov)

    # Precompute the coefficient
    coeff = update_switch * sqrt(lr_cov * (2.0 - lr_cov) * mu_eff)

    # Loop over the latent mean step elements
    for index, step in enumerate(latent_mean_step):

        #
        path_cov[index, 0] += coeff * step

    # Update the mean
    mean += (lr_mean * sigma) * elite_mean_step

    # Update the step size
    sigma *= exp(
        (lr_sigma / damp_sigma) * (ps_norm / expected_path_length - 1))

    # Check if the updated sigma is lower than 1e-15
    if sigma < 1e-15:

        # Clip to 1e-15
        sigma = 1e-15

    # Else, check if the updated sigma is above 1.0
    elif sigma > 1.0:

        # Clip to 1.0
        sigma = 1.0

    return sigma, elite_indices
