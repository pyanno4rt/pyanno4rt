"""Naive Bayes tune grid."""

# Author: Tim Ortkamp

# %% External package import

from functools import partial
from itertools import product

# %% Internal package import

from pyanno4rt.tools import filter_dict
from pyanno4rt.validation import validate_item, validate_subtype, validate_type

# %% Class definition


class TuneGridNB():
    """
    Naive Bayes tune grid class.

    This class provides methods to set, validate and serialize a \
    hyperparameter tune grid for a naive Bayes model.

    Parameters
    ----------
    priors : None or list, default=None
        Options for the prior probabilities of the classes.

    var_smoothing : None or list, default=None
        Options for the portion of the largest variance of all features added \
        to variances for calculation stability.

    .. note:: If arguments are passed as None, default values will be applied.

    Attributes
    ----------
    priors : list
        See 'Parameters'.

    var_smoothing : list
        See 'Parameters'.
    """

    def __init__(
            self,
            priors=None,
            var_smoothing=None):

        # Get the input arguments
        arguments = filter_dict(vars(), remove_keys=('self',))

        # Set the defaults
        defaults = {
            'priors': [None] + [[i/10, 1-i/10] for i in range(1, 10)],
            'var_smoothing': [10**i for i in range(-9, 0)]
            }

        # Update the input arguments
        arguments = {
            key: value if value is not None else defaults[key]
            for key, value in arguments.items()}

        # Validate the input arguments
        self.validate(arguments)

        # Loop over the input arguments
        for item in arguments.items():

            # Set the attribute
            setattr(self, *item)

    def to_dict(self):
        """Serialize the tune grid into a dictionary."""

        return vars(self)|{'name': 'Naive Bayes'}

    @classmethod
    def from_dict(
            cls,
            dictionary):
        """
        Deserialize the tune grid from a dictionary.

        Parameters
        ----------
        dictionary : dict
            Dictionary with the tune grid parameters.

        Returns
        -------
        object of class \
            :class:`~pyanno4rt.learning.tuning.grids._tune_grid_nb.TuneGridNB`
            The object used to handle the tune grid parameters.
        """

        return cls(**dictionary)

    def to_list(self):
        """
        Get the grid search proposals.

        Returns
        -------
        list
            Grid search proposals.
        """

        # Set the parameter keys
        keys = ('priors', 'var_smoothing')

        # Set the parameter values
        values = list(product(self.priors, self.var_smoothing))

        return [dict(zip(keys, value)) for value in values]

    def validate(
            self,
            inputs):
        """
        Validate the input arguments.

        Parameters
        ----------
        inputs : dict
            Dictionary with the mappings between argument names and values.
        """

        # Get the validation map
        validation_map = {
            'priors': (
                partial(validate_type, options=list),
                partial(validate_subtype, options=(list, type(None)))
                ),
            'var_smoothing': (
                partial(validate_type, options=list),
                partial(validate_subtype, options=(int, float)),
                partial(validate_item, reference=0, sign='>')
                )
            }

        # Loop over the dictionary items
        for key, value in inputs.items():

            # Loop over the validation functions
            for function in validation_map[key]:

                # Run the validation function
                function(key, value)
