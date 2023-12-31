"""Patient sex feature."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% Internal package import

from pyanno4rt.learning_model.features.catalogue import DemographicFeature

# %% Class definition


class PatientSex(DemographicFeature):
    """Patient sex feature class."""

    @staticmethod
    def compute(value):
        """Get the sex."""
        # Convert the string value into a binary value
        sex = {'male': 1, 'female': 0}

        return sex[value]
