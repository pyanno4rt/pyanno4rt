"""Random forest tune space."""

# Author: Tim Ortkamp

# %% Internal package import

from pyanno4rt.checking import (
    check_length, check_string_is_number, check_subtype, check_type,
    check_value, check_value_in_set)
from pyanno4rt.tools import filter_dict

# %% Class definition


class TuneSpaceRF():
    """
    Random forest tune space class.

    This class provides methods to set, validate and serialize a \
    hyperparameter tune space for the random forest.

    Parameters
    ----------
    

    Attributes
    ----------
    
    - 'n_estimators' : number of trees in the forest
    - 'criterion' : measure for the quality of a split
    - 'max_depth' : maximum depth of the tree
    - 'min_samples_split' : minimum number of samples required to \
        split an internal node
    - 'min_samples_leaf' : minimum number of samples required at a \
        leaf node
    - 'min_weight_fraction_leaf' : minimum weighted fraction of the \
        sum of weights required at each node
    - 'max_features' : number of features considered at each split
    - 'bootstrap' : indicator for the use of bootstrap samples to \
        build the trees
    - 'class_weight' : weights associated with the classes
    - 'ccp_alpha' : complexity parameter for minimal cost-complexity \
        pruning
    """

    def __init__(
            self,
            ):

        # Get the input arguments
        inputs = filter_dict(vars(), remove_keys=('self',))

        # Check the input arguments
        self.check(inputs)

        # Loop over the input arguments
        for item in inputs.items():

            # Set the attribute
            setattr(self, *item)

    def to_dict(self):
        """Serialize the tune space into a dictionary."""

        return {'tune_space': vars(self)}

    @classmethod
    def from_dict(
            cls,
            dictionary):
        """
        Deserialize the tune space from a dictionary.

        Parameters
        ----------
        dictionary : dict
            Dictionary with the tune space parameters.

        Returns
        -------
        object of class \
            :class:`~pyanno4rt.learning.forest._tune_space_rf.TuneSpaceRF`
            The object used to handle the tune space parameters.
        """

        return cls(**dictionary)

    def check(
            self,
            inputs):
        """
        Check the input arguments.

        Parameters
        ----------
        inputs : dict
            Dictionary with the mappings between argument names and values.
        """

        # Get the check map
        check_map = {}

        # Loop over the dictionary items
        for key, value in inputs.items():

            # Loop over the check functions
            for function in check_map[key]:

                # Run the check function
                function(key, value)
