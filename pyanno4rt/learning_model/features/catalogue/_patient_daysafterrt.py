"""Patient days-after-radiotherapy feature."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% Internal package import

from pyanno4rt.learning_model.features.catalogue import DemographicFeature

# %% Class definition


class PatientDaysafterrt(DemographicFeature):
    """Patient days-after-radiotherapy feature class."""

    @staticmethod
    def compute(value):
        """Get the days-after-radiotherapy."""
        return value
