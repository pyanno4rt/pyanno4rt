"""Abstract feature classes."""

# Author: Tim Ortkamp

# %% External package import

from abc import ABCMeta, abstractmethod

# %% Class definition


class DosiomicFeature(metaclass=ABCMeta):
    """Abstract superclass for dosiomic features."""

    # Set the feature class
    feature_class = 'Dosiomics'

    # Initialize the feature value and gradient functions
    value_function = None
    gradient_function = None

    # Initialize the boolean indicators for one-time jitting of the functions
    value_is_jitted = False
    gradient_is_jitted = False

    @abstractmethod
    def compute(
            self,
            dose,
            *args):
        """Abstract method for computing the feature value."""

    @abstractmethod
    def differentiate(
            self,
            dose,
            *args):
        """Abstract method for differentiating the feature."""


class RadiomicFeature(metaclass=ABCMeta):
    """Abstract superclass for radiomic features."""

    # Set the feature class
    feature_class = 'Radiomics'

    @abstractmethod
    def compute(
            self,
            mask,
            spacing):
        """Abstract method for computing the feature value."""
