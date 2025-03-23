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
        """Abstract method for computing the feature value."""

    @staticmethod
    @abstractmethod
    def differentiate(
            dose,
            *args):
        """Abstract method for computing the feature gradient."""


class RadiomicFeature(metaclass=ABCMeta):
    """Abstract superclass for radiomic features."""

    # Set the feature class name
    feature_class = 'Radiomics'

    @staticmethod
    @abstractmethod
    def compute(
            mask,
            spacing):
        """Abstract method for computing the feature value."""
