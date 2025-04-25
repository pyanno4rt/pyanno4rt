"""Evaluation metrics display options."""

# Author: Tim Ortkamp

# %% External package import

from functools import partial

# %% Internal package import

from pyanno4rt.checking import check_type, check_value_in_set
from pyanno4rt.tools import filter_dict

# %% Class definition


class DisplayOptions():
    """
    Evaluation metrics display options class.

    This class provides methods to set, validate and serialize the display \
    options for the evaluation metrics of the learning models.

    Parameters
    ----------
    graphs : None or list, default=None
        Options ('AUC-ROC', 'AUC-PR', 'F1') for the model evaluation graphs.

    kpis : None or list, default=None
        Options ('Logloss', 'Brier score', 'Subset accuracy', 'Cohen Kappa',
        'Hamming loss', 'Jaccard score', 'Precision', 'Recall', 'F1 score',
        'MCC', 'AUC') for the model evaluation KPIs.

    .. note:: If arguments are passed as None, default values will be applied.

    Attributes
    ----------
    graphs : None or list
        See 'Parameters'.

    kpis : None or list
        See 'Parameters'.
    """

    def __init__(
            self,
            graphs=None,
            kpis=None):

        # Get the input arguments
        inputs = filter_dict(vars(), remove_keys=('self',))

        # Set the default argument values
        defaults = {
            'graphs': ['AUC-ROC', 'AUC-PR', 'F1'],
            'kpis': [
                'Logloss', 'Brier score', 'Subset accuracy', 'Cohen Kappa',
                'Hamming loss', 'Jaccard score', 'Precision', 'Recall',
                'F1 score', 'MCC', 'AUC']}

        # Update the input arguments with the defaults, if applicable
        inputs = {
            key: value if value is not None else defaults[key]
            for key, value in inputs.items()}

        # Check the input arguments
        self.check(inputs)

        # Loop over the input arguments
        for item in inputs.items():

            # Set the attribute
            setattr(self, *item)

    def to_dict(self):
        """Serialize the display options into a dictionary."""

        return vars(self)

    @classmethod
    def from_dict(
            cls,
            dictionary):
        """
        Deserialize the display options from a dictionary.

        Parameters
        ----------
        dictionary : dict
            Dictionary with the display options.

        Returns
        -------
        object of class \
            :class:`~pyanno4rt.learning.evaluation._display_options.DisplayOptions`
            The object used to handle the display options.
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
        check_map = {
            'graphs': (
                partial(check_type, options=list),
                partial(check_value_in_set, options=(
                    'AUC-ROC', 'AUC-PR', 'F1'))),
            'kpis': (
                partial(check_type, options=list),
                partial(check_value_in_set, options=(
                    'Logloss', 'Brier score', 'Subset accuracy',
                    'Cohen Kappa', 'Hamming loss', 'Jaccard score',
                    'Precision', 'Recall', 'F1 score', 'MCC', 'AUC')))}

        # Loop over the dictionary items
        for key, value in inputs.items():

            # Loop over the check functions
            for function in check_map[key]:

                # Run the check function
                function(key, value)
