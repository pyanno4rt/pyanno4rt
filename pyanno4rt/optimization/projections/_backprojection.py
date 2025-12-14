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
    __dose__ : ndarray
        Current (cached) dose vector.

    __dose_gradient__ : ndarray
        Current (cached) dose gradient.

    __fluence__ : ndarray
        Current (cached) fluence vector.

    __fluence_gradient__ : ndarray
        Current (cached) fluence gradient.
    """

    def __init__(self):

        # Initialize the dose, dose gradient, fluence and fluence gradient
        self.__dose__ = array([])
        self.__dose_gradient__ = array([])
        self.__fluence__ = array([])
        self.__fluence_gradient__ = array([])

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
        if not array_equal(self.__fluence__, fluence):

            # Update the cached dose vector
            self.__dose__ = self.compute_dose_result(fluence)

            # Update the cached fluence
            self.__fluence__ = copy(fluence)

        return self.__dose__

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
        if not array_equal(self.__dose_gradient__, dose_gradient):

            # Update the cached fluence gradient
            self.__fluence_gradient__ = self.compute_fluence_gradient_result(
                dose_gradient)

            # Update the cached dose gradient
            self.__dose_gradient__ = copy(dose_gradient)

        return self.__fluence_gradient__

    def get_dose(self):
        """
        Get the dose vector.

        Returns
        -------
        ndarray
            Dose vector.
        """

        return self.__dose__

    def get_fluence_gradient(self):
        """
        Get the fluence gradient.

        Returns
        -------
        ndarray
            Fluence gradient.
        """

        return self.__fluence_gradient__

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
