"""Backprojection."""

# Author: Tim Ortkamp

# %% External package import

from abc import abstractmethod
from numpy import array, array_equal, copy

# %% Class definition


class Backprojection():
    """
    Backprojection superclass.

    This class provides a caching system, methods to get/compute dose and \
    fluence gradient, and abstract methods to implement projection rules.

    Attributes
    ----------
    _dose : ndarray
        Cached dose vector.

    _dose_gradient : ndarray
        Cached dose gradient.

    _fluence : ndarray
        Cached fluence vector.

    _fluence_gradient : ndarray
        Cached fluence gradient.
    """

    def __init__(self):

        # Initialize the dose, dose gradient, fluence and fluence gradient
        self._dose = array([])
        self._dose_gradient = array([])
        self._fluence = array([])
        self._fluence_gradient = array([])

    def compute_dose(
            self,
            fluence):
        """
        Compute the dose vector from the fluence vector.

        Parameters
        ----------
        fluence : ndarray
            Fluence vector.

        Returns
        -------
        ndarray
            Dose vector.
        """

        # Check if the cached fluence does not resemble the input
        if not array_equal(self._fluence, fluence):

            # Update the cached dose vector
            self._dose = self.compute_dose_result(fluence)

            # Update the cached fluence
            self._fluence = copy(fluence)

        return self._dose

    def compute_fluence_gradient(
            self,
            dose_gradient):
        """
        Compute the fluence gradient from the dose gradient.

        Parameters
        ----------
        dose_gradient : ndarray
            Dose gradient.

        Returns
        -------
        ndarray
            Fluence gradient.
        """

        # Check if the cached dose gradient does not resemble the input
        if not array_equal(self._dose_gradient, dose_gradient):

            # Update the cached fluence gradient
            self._fluence_gradient = self.compute_fluence_gradient_result(
                dose_gradient)

            # Update the cached dose gradient
            self._dose_gradient = copy(dose_gradient)

        return self._fluence_gradient

    def get_dose(self):
        """
        Get the dose vector.

        Returns
        -------
        ndarray
            Dose vector.
        """

        return self._dose

    def get_fluence_gradient(self):
        """
        Get the fluence gradient.

        Returns
        -------
        ndarray
            Fluence gradient.
        """

        return self._fluence_gradient

    @abstractmethod
    def compute_dose_result(
            self,
            fluence):
        """
        Compute the dose vector from the fluence vector.

        Parameters
        ----------
        fluence : ndarray
            Fluence vector.

        Returns
        -------
        ndarray
            Dose vector.
        """

    @abstractmethod
    def compute_fluence_gradient_result(
            self,
            dose_gradient):
        """
        Compute the fluence gradient from the dose gradient.

        Parameters
        ----------
        dose_gradient : ndarray
            Dose gradient.

        Returns
        -------
        ndarray
            Fluence gradient.
        """
