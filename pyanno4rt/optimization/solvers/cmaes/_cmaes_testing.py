"""Covariance matrix adaptation evolution strategy (CMA-ES)."""

# Author: Tim Ortkamp

# %% External package import

from time import time

from collections import deque
from math import inf
from numpy import (
    arange, argmin, argpartition, array, diag, einsum, exp, eye, log, maximum,
    median, minimum, sqrt, zeros)
from numpy import sum as nsum
from numpy.linalg import inv, norm, svd, svdvals
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
        self._path_sigma = zeros(self._number_of_variables)
        self._path_cov = zeros(self._number_of_variables)
        self._mean = zeros(self._number_of_variables)
        self._cov = eye(self._number_of_variables)
        self._sq_cov = eye(self._number_of_variables)

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

        # Perform SVD on the covariance matrix
        left_svec, svals, _ = svd(self._cov)

        # Get the square root of the covariance matrix
        self._sq_cov = left_svec@diag(sqrt(svals))

        # Sample steps from the multivariate Gaussian
        steps = zsamples@self._sq_cov

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
            * inv(self._sq_cov).T
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
        self._cov = (
            (1+(1-update_switch)*self._lr_rank_1*self._lr_cov*(2-self._lr_cov))
            * self._cov
            + self._lr_rank_1*(self._path_cov@self._path_cov.T-self._cov)
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

            ########################################################
            svs = svdvals(self._cov)
            update_plot(self._opt_iter, self._mean, self._cov, svs)
            ########################################################

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

# %% Toy problem class

from math import pi
from numpy import cos, sin

class ToyProblem:
    """
    A toy problem class to test the CMA algorithm with different benchmark \
    functions.

    Parameters
    ----------
    name : {'rastrigin', 'sphere', 'styblinski-tang'}
        Name of the test function.

    initial_x : ndarray
        Initial decision variables.

    Attributes
    ----------
    name : {'rastrigin', 'sphere', 'styblinski-tang'}
        See 'Parameters'.

    f : function
        Objective function.

    g : function
        Gradient function.

    variable_bounds : tuple
        Tuple with two lists for the lower and upper variable bounds.
    """

    def __init__(
            self,
            name,
            x0):

        # Get the function name
        self.name = name
        self.x0 = x0

        # Map the names
        functions = {
            'rastrigin': self.rastrigin,
            'sphere': self.sphere,
            'styblinski-tang': self.styblinski_tang}

        # Get the objective and gradient functions
        self.f, self.g, self.variable_bounds = functions[name]()

    def objective(self, x):
        """
        Compute the objective.

        Parameters
        ----------
        x : ndarray
            Decision vector.

        Returns
        -------
        int or float
            Objective value.
        """

        return self.f(x)

    def gradient(
            self,
            x):
        """
        Compute the gradient.

        Parameters
        ----------
        x : ndarray
            Decision vector.

        Returns
        -------
        ndarray
            Gradient.
        """

        return self.g(x)

    def rastrigin(self):
        """
        Rastrigin function.

        Global minimum: f(0,...,0) = 0.0, search domain: [-5.12, 5.12].
        """

        def f(x):
            return 10*len(x) + sum(x**2 - 10*cos(2*pi*x))

        def g(x):
            return 2*x+20*pi*sin(2*pi*x)

        bounds = ([-5.12]*len(self.x0), [5.12]*len(self.x0))

        return f, g, bounds

    def sphere(self):
        """
        Sphere function.

        Global minimum: f(0,...,0) = 0.0, search domain: [-inf, inf].
        """

        def f(x):
            return sum(x**2)

        def g(x):
            return 2*x

        bounds = ([-inf]*len(self.x0), [inf]*len(self.x0))

        return f, g, bounds

    def styblinski_tang(self):
        """
        Styblinski-Tang function.

        Global minimum: -39.16617*len(x) < f(-2.903534,...,-2.903534) \
        < -39.16616*len(x), search domain: [-5, 5].
        """

        def f(x):
            return sum(x**4-16*x**2+5*x)/2

        def g(x):
            return 2*x**3-16*x+2.5

        bounds = ([-5]*len(self.x0), [5]*len(self.x0))

        return f, g, bounds


# %% Test run

# Set the initial vector
initial_x = array([5]*2)

# Initialize the toy problem
prob = ToyProblem(
    name='styblinski-tang',
    x0=initial_x)

# Initialize the CMA solver
solver = CMAES(
    number_of_variables=len(initial_x),
    objective=prob.f,
    lower_variable_bounds=array(prob.variable_bounds[0]),
    upper_variable_bounds=array(prob.variable_bounds[1]),
    gradient=prob.g,
    number_of_individuals=100,
    initial_sigma=2.0,
    maximum_iterations=200,
    maximum_wall_time=7200,
    fitness_threshold=None,
    fitness_window_size=30,
    tolerance=1e-3,
    callback=None)

# Optimize the variables
result = solver.optimize(initial_x)

# Get the iteration-wise eigenvalues
sv = solver._singular_values

# %% Plot covariance matrix over function

import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

def plot_iter_sv(svals, iteration, fname, k):
    """Plot the singular values for a fixed iteration."""

    # Plotting on a semi-log scale (y-axis is logarithmic)
    plt.figure(figsize=(10, 6))

    #
    values = svals[iteration][:k]

    # Plot the singular values
    plt.semilogy(values, marker='o', linestyle='-', color='b')

    plt.title(f'{fname} (iteration {iteration})', fontweight='bold')
    plt.ylabel('Singular Value ($\sigma_i$) (log scale)')
    plt.xlabel('Singular Value Index')
    # plt.xticks([i for i in range(len(values))])
    plt.grid(True, which="both", ls="--", color='0.7')
    plt.savefig(f'/home/tim/Downloads/{fname}_{iteration}.pdf')
    plt.show()

plot_iter_sv(sv, 0, prob.name, 20)
plot_iter_sv(sv, len(sv)//2, prob.name, 20)
plot_iter_sv(sv, len(sv)-1, prob.name, 20)

def plot_sv_paths(svals, fname, space):
    """Plot the iteration-wise singular value paths."""

    # Plotting on a semi-log scale (y-axis is logarithmic)
    plt.figure(figsize=(10, 6))

    # Plot the singular values
    for values in zip(*svals):

        #
        subvalues = values[::space]

        #
        plt.semilogy(
            array(range(len(subvalues)))*space, subvalues, marker='.',
            linestyle='-', color='b')

    plt.title(f'{fname}', fontweight='bold')
    plt.ylabel('Singular Value ($\sigma_i$) (log scale)')
    plt.xlabel('Optimization iteration')
    plt.grid(True, which="both", ls="--", color='0.7')
    plt.savefig(f'/home/tim/Downloads/{fname}.pdf')
    plt.show()

plot_sv_paths(sv, prob.name, 1)

# Interactive plotting
plt.ion()

# Create plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=30, azim=-45)

def update_plot(iteration, mu_2d, cov_2d, svs):

    # Clear the axis
    ax.clear()

    # Create meshgrid
    x_range = np.linspace(-5, 5, 50)
    y_range = np.linspace(-5, 5, 50)
    X, Y = np.meshgrid(x_range, y_range)

    # Plot function as surface
    Z = prob.f(array([X, Y]))
    ax.plot_surface(
        X, Y, Z, cmap='viridis', alpha=0.3, antialiased=True, zorder=5)

    # Plot mean vector
    mu_z = prob.f(array([mu_2d[0], mu_2d[1]]))
    ax.text(mu_2d[0], mu_2d[1], mu_z, "●", color='orange', fontsize=14,
            ha='center', va='center', zorder=100)

    # Compute the circle
    vals, vecs = np.linalg.eig(cov_2d)
    t = np.linspace(0, 2*np.pi, 100)
    circle = np.stack([np.cos(t), np.sin(t)])

    # Transform into ellipsoid
    scaling = 3 * np.sqrt(np.maximum(vals, 0))
    ellipse_points = (vecs @ (scaling[:, None] * circle))

    # Compute the center of the ellipsoid
    ell_x = ellipse_points[0, :] + mu_2d[0]
    ell_y = ellipse_points[1, :] + mu_2d[1]

    # Compute the z-level of the ellipsoid
    ell_z = [mu_z]*100

    # Plot covariance matrix
    ax.plot(
        ell_x, ell_y, ell_z, color='red', linewidth=3, label="3σ ellipse",
        zorder=5)

    # Add the mean and singular values
    s_vals = np.sqrt(np.maximum(vals, 0))
    label_txt = (
        f"mean: {(round(float(mu_2d[0]), 4), round(float(mu_2d[1]), 4))}"
        f"\nλ1: {s_vals[0]:.2f}\nλ2: {s_vals[1]:.2f}")
    ax.text(
        mu_2d[0], mu_2d[1], mu_z+10,
        label_txt, zorder=10,
        bbox=dict(
            facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title(f"Iteration: {iteration+1}")
    plt.draw()
    plt.pause(3)
    plt.show()

plt.pause(10)
plt.close()
