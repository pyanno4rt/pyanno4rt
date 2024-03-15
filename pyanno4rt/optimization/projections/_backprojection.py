"""Backprojection."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from abc import abstractmethod
from numpy import array, array_equal

# %% Class definition


class BackProjection():
    """
    Backprojection superclass.

    This class provides caching attributes, methods to get/compute the dose \
    and the fluence gradient, and abstract methods to implement projection \
    rules within the inheriting classes.

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
        Compute the dose vector from the fluence and update the cache.

        Parameters
        ----------
        fluence : ndarray
            Values of the fluence vector.

        Returns
        -------
        ndarray
            Values of the dose vector.
        """

        # Check if the cached fluence does not resemble the input
        if not array_equal(self.__fluence__, fluence):

            # Compute the dose vector from the fluence
            self.__dose__ = self.compute_dose_result(fluence)

            # Update the cached fluence
            self.__fluence__ = fluence

        return self.__dose__

    def compute_fluence_gradient(
            self,
            dose_gradient):
        """
        Compute the fluence gradient from the dose gradient and update the \
        cache.

        Parameters
        ----------
        dose_gradient : ndarray
            Values of the dose gradient.

        Returns
        -------
        ndarray
            Values of the fluence gradient.
        """

        # Check if the cached dose gradient does not resemble the input
        if not array_equal(self.__dose_gradient__, dose_gradient):

            # Compute the fluence gradient from the dose gradient
            self.__fluence_gradient__ = self.compute_fluence_gradient_result(
                dose_gradient)

            # Update the cached dose gradient
            self.__dose_gradient__ = dose_gradient

        return self.__fluence_gradient__

    def get_dose(self):
        """
        Get the dose vector.

        Returns
        -------
        ndarray
            Values of the dose vector.
        """

        return self.__dose__

    def get_fluence_gradient(self):
        """
        Get the fluence gradient.

        Returns
        -------
        ndarray
            Values of the fluence gradient.
        """

        return self.__fluence_gradient__

    @abstractmethod
    def compute_dose_result(
            self,
            fluence):
        """
        Compute the dose projection from the fluence vector.

        Parameters
        ----------
        fluence : ndarray
            Values of the fluence vector.

        Returns
        -------
        ndarray
            Values of the dose vector.
        """

    @abstractmethod
    def compute_fluence_gradient_result(
            self,
            dose_gradient):
        """
        Compute the fluence gradient projection from the dose gradient.

        Parameters
        ----------
        dose_gradient : ndarray
            Values of the dose gradient.

        Returns
        -------
        ndarray
            Values of the fluence gradient.
        """
