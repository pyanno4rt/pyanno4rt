"""Covariance matrix adaptation evolutionary strategy (CMA-ES)."""

# Author: Tim Ortkamp
# Adapted from Nomura & Shibata (2024): https://arxiv.org/pdf/2402.01373

# %% External package import

from time import time

from math import inf, pi
from numpy import (
    argmin, argpartition, around, array, clip, cos, exp, identity, log, sin,
    zeros)
from numpy.linalg import cholesky, eig, matrix_power, norm
from numpy.random import multivariate_normal, seed

# %% Covariance matrix adaptation evolutionary algorithm


class CMA:
    """
    Covariance matrix adaptation evolutionary strategy class.

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
            x0,
            problem,
            maximum_iterations,
            tolerance,
            early_stopping_rounds
            ):

        # Get the properties of the optimization problem
        self._xdim = len(x0)
        self.fitness = problem.objective
        self.gradient = getattr(problem, 'gradient', None)
        self.x0 = problem.x0
        self.lower_bound = array(problem.variable_bounds[0])
        self.upper_bound = array(problem.variable_bounds[1])

        # Get the stopping criteria
        self.maximum_iterations = maximum_iterations
        self.tolerance = tolerance
        self.early_stopping_threshold = tolerance
        self.early_stopping_rounds = early_stopping_rounds
        self.early_stopping_iter = 0

        # Calculate the algorithm parameters
        self._lambda = 4 + int(3*log(self._xdim))
        self._mu = int(self._lambda/2)
        self._w = array([
            (log(self._mu+0.5)-log(i))
            / (sum(log(self._mu+0.5)-log(j) for j in range(1, self._mu+1)))
            for i in range(1, self._mu+1)])
        self._mu_eff = 1/sum(self._w)
        self._cs = (self._mu_eff+2)/(self._xdim+self._mu_eff+3)
        self._cc = 4/(self._xdim+4)
        self._c1 = (
            (2*min(1, self._lambda/6))/((self._xdim+1.3)**2+self._mu_eff))
        self._cmu = (
            (2*(self._mu_eff-2+1/self._mu_eff))
            / ((self._xdim+2)**2+self._mu_eff))

        # Initialize the solver variables
        self._iter = 0
        self._sigma = 0.3
        self._ps = zeros(self._xdim)
        self._pc = zeros(self._xdim)
        self._mean = x0
        self._covmat = identity(self._xdim)

        # Initialize the eigenvalues list
        self._eigenvalues = []

        # Initialize the result dictionary
        self._result = {
            'xopt': x0, 'fopt': inf, 'solver_info': '', 'runtime': None}

    def check_termination(self):
        """
        Check the termination criteria.

        Returns
        -------
        bool
            Indicator for termination.
        """

        # Check if the maximum number of iterations has been reached
        if self._iter >= self.maximum_iterations:

            # Add the solver info
            self._result['solver_info'] = 'MAX_ITER_REACHED'

            return True

        # Check if maximum number of early stopping rounds has been reached
        if self.early_stopping_iter >= self.early_stopping_rounds:

            # Add the solver info
            self._result['solver_info'] = 'EARLY_STOPPING'

            return True

        return False

    def optimize(self):
        """Optimize the decision variables."""

        #
        start = time()

        # Continue until termination criteria are fulfilled
        while self.check_termination() is False:

            # Store the eigenvalues of the covariance matrix
            evals, _ = eig(self._covmat)
            self._eigenvalues.append(evals)

            #
            z = multivariate_normal(
                zeros(self._xdim), identity(self._xdim), (self._lambda,))

            #
            y = z@cholesky(self._covmat)

            # Sample the new population
            x = clip(
                self._mean + self._sigma*y, a_min=self.lower_bound,
                a_max=self.upper_bound)

            # Compute the fitness values
            f = array([self.fitness(xs) for xs in x])

            #
            fmin, fdiff = f.min(), self._result['fopt'] - f.min()

            #
            if fdiff > 0:

                #
                self._result['fopt'] = fmin
                self._result['xopt'] = x[argmin(f)]

                #
                self.early_stopping_iter = 0

            else:

                #
                self.early_stopping_iter += 1

            #
            print(f'At iterate {self._iter}: f={around(fmin, 8)}')

            # Get the indices of the k best fitness values
            inds = argpartition(f, self._mu)[:self._mu]

            #
            self._ps = (
                (1-self._cs)*self._ps
                + (self._cs*(2-self._cs)*self._mu_eff)**0.5
                * matrix_power(cholesky(self._covmat), -1)
                * sum(self._w[i]*y[idx] for i, idx in enumerate(inds)))

            #
            mul = self._xdim**0.5*(1-1/(4*self._xdim)+1/(21*self._xdim**2))

            #
            hs = (
                norm(self._ps)
                < (1-(1-self._cs)**(2*(self._iter+1)))**0.5
                * (1.4+2/(self._xdim+1))
                * mul)

            #
            self._pc = (
                (1-self._cc)*self._pc
                + hs*(self._cc*(2-self._cc)*self._mu_eff)**0.5
                * sum(self._w[i]*y[idx] for i, idx in enumerate(inds)))

            # Adapt the mean vector
            self._mean = (
                self._mean + self._sigma
                * sum(self._w[i]*y[idx] for i, idx in enumerate(inds)))

            #
            self._sigma = self._sigma*exp(
                (self._cs/2)*((norm(self._ps)/mul)-1))

            # Adapt the covariance matrix
            self._covmat = (
                (1+(1-hs)*self._c1*self._cc*(2-self._cc))*self._covmat
                + self._c1*(self._pc@self._pc.T-self._covmat)
                + self._cmu*sum(
                    self._w[i]*y[idx]@y[idx].T - self._w[i]*self._covmat
                    for i, idx in enumerate(inds)))

            #
            self._iter += 1

        #
        self._result['runtime'] = time()-start

        #
        print(
            f'CMA-ES solver took {around(self._result["runtime"], 2)} seconds '
            f'for optimization ({self._result["solver_info"]}) ...')
        print(
            'Optimal function value found: '
            f'{around(self._result["fopt"], 4)} ...')

        return self._result


# %% Toy problem class


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
initial_x = array([0]*3)

# Initialize the toy problem
prob = ToyProblem(
    name='sphere',
    x0=initial_x)

# Initialize the CMA solver
solver = CMA(
    x0=initial_x,
    problem=prob,
    maximum_iterations=5000,
    tolerance=1e-3,
    early_stopping_rounds=30)

# Optimize the variables
result = solver.optimize()

# Get the iteration-wise eigenvalues
evs = solver._eigenvalues
