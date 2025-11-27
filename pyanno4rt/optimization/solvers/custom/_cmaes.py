"""Covariance matrix adaptation evolution strategy (CMA-ES)."""

# Author: Tim Ortkamp
# Adapted from Nomura & Shibata (2024): https://arxiv.org/pdf/2402.01373

# %% External package import

from time import time

from math import inf
from numpy import (
    argmin, argpartition, array, exp, eye, log, maximum, minimum, zeros)
from numpy.linalg import cholesky, inv, norm, svdvals
from numpy.random import seed, standard_normal

# %% Covariance matrix adaptation evolution algorithm


class CMA:
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
            callback=None
            ):

        # Get the properties of the optimization problem
        self._xdim = number_of_variables
        self.fitness = objective
        self.gradient = gradient
        self.lower_bound = lower_variable_bounds
        self.upper_bound = upper_variable_bounds

        # Get the stopping criteria
        self.maximum_iterations = maximum_iterations
        self.tolerance = tolerance
        self.early_stopping_threshold = tolerance
        self.early_stopping_rounds = (
            inf if early_stopping_rounds is None else early_stopping_rounds)
        self._early_stopping_iter = 0

        # Get the callback function
        self._callback = callback

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
        self._cm = 1
        self._ds = 1+2*max(0, ((self._mu_eff-1)/self._xdim)**0.5-1)+self._cs
        self._ps = zeros(self._xdim)
        self._pc = zeros(self._xdim)
        self._mean = zeros(self._xdim)
        self._covmat = eye(self._xdim)

        # Initialize the singular values list
        self._svalues = []

        # Initialize the result dictionary
        self._result = {
            'xopt': None, 'fopt': inf, 'solver_info': None, 'runtime': None}

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
        if self._early_stopping_iter >= self.early_stopping_rounds:

            # Add the solver info
            self._result['solver_info'] = 'EARLY_STOPPING'

            return True

        return False

    def ask(self):
        """Ask for the next generation."""

        # Sample from the standard multivariate Gaussian
        z = standard_normal((self._lambda, self._xdim))

        # Calculate the step vectors
        y = z@cholesky(self._covmat)

        # Sample the new population
        x = self._mean + self._cm*self._sigma*y

        return minimum(maximum(x, self.lower_bound), self.upper_bound), y

    def optimize(
            self,
            x0):
        """
        Optimize the decision variables.

        Returns
        -------
        dict
            Dictionary with the optimization results.
        """

        # Start the runtime recording
        start = time()

        # Initialize the mean value
        self._mean = x0

        # Compute the expected path length
        mu_p = self._xdim**0.5*(1-1/(4*self._xdim)+1/(21*self._xdim**2))

        # Continue until termination criteria are fulfilled
        while self.check_termination() is False:

            # Store the eigenvalues of the covariance matrix
            self._svalues.append(svdvals(self._covmat))

            # "Ask" for the new population
            x, y = self.ask()

            # Compute the fitness values
            f = array([self.fitness(xs) for xs in x])

            # Get the minimum fitness and the fitness improvement
            fmin, fdiff = f.min(), self._result['fopt'] - f.min()

            # Check if a better solution has been found
            if fdiff > 0:

                # Update the optimal value and point
                self._result['fopt'] = fmin
                self._result['xopt'] = x[argmin(f)]

                # Check if the minimum improvement has been reached
                if fdiff >= self.early_stopping_threshold:

                    # Reset the early stopping counter
                    self._early_stopping_iter = 0

            # Check if improvement alone is insufficient
            elif (self.early_stopping_rounds == inf
                    and fdiff < self.tolerance):

                # Add the solver info and stop
                self._result['solver_info'] = 'NO_IMPROV'
                break

            else:

                # Increment the early stopping counter
                self._early_stopping_iter += 1

            # Check if a callback has been provided
            if self._callback is not None:

                # Print a message about the current optimal values
                self._callback(self._result)

            # Get the indices of the k best fitness values
            inds = argpartition(f, self._mu)[:self._mu]

            # Compute the weighted sum of the best step vectors
            yw = self._w@y[inds,:]

            # Update the step-size evolution path
            self._ps = (
                (1-self._cs)*self._ps
                + (self._cs*(2-self._cs)*self._mu_eff)**0.5
                * cholesky(inv(self._covmat)) * yw)

            # Precompute the Heaviside function for self._pc
            hs = (
                norm(self._ps)
                < (1-(1-self._cs)**(2*(self._iter+1)))**0.5
                *(1.4+2/(self._xdim+1))*mu_p)

            # Update the rank-1 evolution path
            self._pc = (
                (1-self._cc)*self._pc
                + hs*(self._cc*(2-self._cc)*self._mu_eff)**0.5 * yw)

            # Update the mean vector
            self._mean = self._mean + self._sigma*yw

            # Update the step size
            self._sigma = (
                self._sigma*exp((self._cs/self._ds)*((norm(self._ps)/mu_p)-1)))

            # Update the covariance matrix
            self._covmat = (
                (1+(1-hs)*self._c1*self._cc*(2-self._cc))*self._covmat
                + self._c1*(self._pc@self._pc.T-self._covmat)
                + self._cmu*sum(
                    self._w[i]*y[idx]@y[idx].T - self._w[i]*self._covmat
                    for i, idx in enumerate(inds)))

            # Increment the iteration counter
            self._iter += 1

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
# initial_x = array([0]*50)

# # Initialize the toy problem
# prob = ToyProblem(
#     name='rastrigin',
#     x0=initial_x)

# # Initialize the CMA solver
# solver = CMA(
#     number_of_variables=len(initial_x),
#     objective=prob.f,
#     lower_variable_bounds=array(prob.variable_bounds[0]),
#     upper_variable_bounds=array(prob.variable_bounds[1]),
#     gradient=prob.g,
#     maximum_iterations=10000,
#     tolerance=1e-3,
#     early_stopping_rounds=100)

# # Optimize the variables
# result = solver.optimize(x0=initial_x)

# # Get the iteration-wise eigenvalues
# sv = solver._svalues
