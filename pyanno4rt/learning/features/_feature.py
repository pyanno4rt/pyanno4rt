"""Abstract feature classes."""

# Author: Tim Ortkamp

# %% External package import

from abc import ABCMeta, abstractmethod

# %% Class definition


class DosiomicFeature(metaclass=ABCMeta):
    """Abstract superclass for dosiomic features."""

    # Set the feature class name
    feature_class = 'Dosiomics'

    # Initialize the feature value and gradient functions
    value_function = None
    gradient_function = None

    # Initialize the boolean indicators for one-time jitting
    value_is_jitted = False
    gradient_is_jitted = False

    @staticmethod
    @abstractmethod
    def compute(
            dose,
            *args):
        """Compute the feature value."""

    @staticmethod
    @abstractmethod
    def differentiate(
            dose,
            *args):
        """Compute the feature gradient."""


class RadiomicFeature(metaclass=ABCMeta):
    """Abstract superclass for radiomic features."""

    # Set the feature class name
    feature_class = 'Radiomics'

    @staticmethod
    @abstractmethod
    def compute(
            mask,
            resolution):
        """Compute the feature value."""
