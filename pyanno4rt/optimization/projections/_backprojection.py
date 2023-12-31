"""Backprojection."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from abc import abstractmethod
from numpy import array, array_equal

# %% Class definition


class BackProjection():
    """
    Backprojection superclass.

    This class serves as an abstract superclass for different types of \
    projections between the fluence and the dose. It provides a caching \
    system and methods to get/compute the dose and the fluence gradient.

    Attributes
    ----------
    __fluence_cache__ : ndarray
        Cache array for the fluence vector.

    __dose__ : ndarray
        Dose vector computed from ``__fluence_cache__``.

    __dose_gradient_cache__ : ndarray
        Cache array for the dose gradient.

    __fluence_gradient__ : ndarray
        Fluence gradient computed from ``__dose_gradient_cache__``.
    """

    def __init__(self):

        # Initialize the instance attributes
        self.__fluence_cache__ = array([])
        self.__dose__ = array([])
        self.__dose_gradient_cache__ = array([])
        self.__fluence_gradient__ = array([])

    def compute_dose(
            self,
            fluence):
        """
        Return the (cached) dose vector.

        Parameters
        ----------
        fluence : ndarray
            Values of the fluence.

        Returns
        -------
        ndarray
            Values of the dose.
        """
        # Check if the passed fluence is not equal to the cached fluence
        if not array_equal(self.__fluence_cache__, fluence):

            # Compute the dose vector from the fluence
            self.__dose__ = self.compute_dose_result(fluence)

            # Update the cached fluence
            self.__fluence_cache__ = fluence

        return self.__dose__

    def compute_fluence_gradient(
            self,
            dose_gradient):
        """
        Return the (cached) fluence gradient.

        Parameters
        ----------
        dose_gradient : ndarray
            Values of the dose derivatives.

        fluence : ndarray
            Values of the fluence.

        Returns
        -------
        ndarray
            Values of the fluence derivatives.
        """
        # Check if the passed dose gradient is not equal to the cached gradient
        if not array_equal(self.__dose_gradient_cache__, dose_gradient):

            # Compute the fluence gradient from the dose gradient
            self.__fluence_gradient__ = self.compute_fluence_gradient_result(
                dose_gradient)

            # Update the cached dose gradient
            self.__dose_gradient_cache__ = dose_gradient

        return self.__fluence_gradient__

    def get_dose(self):
        """
        Get the dose vector.

        Returns
        -------
        ndarray
            Values of the dose.
        """
        return self.__dose__

    def get_fluence_gradient(self):
        """
        Get the fluence gradient.

        Returns
        -------
        ndarray
            Values of the fluence derivatives.
        """
        return self.__fluence_gradient__

    @abstractmethod
    def compute_dose_result(
            self,
            fluence):
        """
        Compute the dose from the fluence.

        Parameters
        ----------
        fluence : ndarray
            Values of the fluence.

        Returns
        -------
        ndarray
            Values of the dose.
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
            Values of the dose derivatives.

        fluence : ndarray
            Values of the fluence.

        Returns
        -------
        ndarray
            Values of the fluence derivatives.
        """
