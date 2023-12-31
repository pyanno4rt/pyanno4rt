"""Patient age feature."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% Internal package import

from pyanno4rt.learning_model.features.catalogue import DemographicFeature

# %% Class definition


class PatientAge(DemographicFeature):
    """Patient age feature class."""

    @staticmethod
    def compute(value):
        """Get the age."""
        return value
