"""Logistic regression tune space."""

# Author: Tim Ortkamp

# %% External package import

from functools import partial
from hyperopt.hp import choice, uniform

# %% Internal package import

from pyanno4rt.tools import filter_dict
from pyanno4rt.validation import (
    validate_item, validate_item_in_set, validate_length, validate_subtype,
    validate_type)

# %% Class definition


class TuneSpaceLR():
    """
    Logistic regression tune space class.

    This class provides methods to set, validate and serialize a \
    hyperparameter tune space for a logistic regression model.

    Parameters
    ----------
    penalty : None or list, default=None
        Options ('l1', 'l2', 'elasticnet') for the norm of the penalty \
        function.

    C : None or list, default=None
        Range for the inverse of the regularization strength.

    tol : None or list, default=None
        Options for the stopping criteria tolerance.

    class_weight : None or list, default=None
        Options (None, 'balanced') for the weights associated with the classes.

    .. note:: If arguments are passed as None, default values will be applied.

    Attributes
    ----------
    penalty : list
        See 'Parameters'.

    C : list
        See 'Parameters'.

    tol : list
        See 'Parameters'.

    class_weight : list
        See 'Parameters'.
    """

    def __init__(
            self,
            penalty=None,
            C=None,
            tol=None,
            class_weight=None):

        # Get the input arguments
        arguments = filter_dict(vars(), remove_keys=('self',))

        # Set the defaults
        defaults = {
            'penalty': ['l1', 'l2', 'elasticnet'],
            'C': [0.001, 100],
            'tol': [1e-3, 1e-4, 1e-5],
            'class_weight': [None, 'balanced']
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
        """Serialize the tune space into a dictionary."""

        return vars(self)|{'name': 'Logistic Regression'}

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
            :class:`~pyanno4rt.learning.tuning.spaces._tune_space_lr.TuneSpaceLR`
            The object used to handle the tune space parameters.
        """

        return cls(**dictionary)

    def to_space(self):
        """
        Get the search space.

        Returns
        -------
        dict
            Dictionary with the search intervals.
        """

        return {
            'regularization': choice(
                'regularization', [
                    {'penalty': None,
                     'solver': choice(
                         'solver_None',
                         ['lbfgs', 'newton-cg', 'newton-cholesky', 'sag'])},
                    *[{'penalty': norm,
                       'solver': choice(
                           f'solver_{norm}',
                           ['liblinear', 'saga'] if norm == 'l1'
                           else [
                               'lbfgs', 'liblinear', 'newton-cg',
                               'newton-cholesky', 'sag', 'saga']),
                       'C': uniform(f'C_{norm}', self.C[0], self.C[1])
                       }
                      if norm != 'elasticnet' else
                      {'penalty': 'elasticnet',
                       'l1_ratio': 0.5,
                       'solver': choice(f'solver_{norm}', ['saga']),
                       'C': uniform(f'C_{norm}', self.C[0], self.C[1])
                       }
                      for norm in self.penalty]
                    ]),
            'tol': choice('tol', self.tol),
            'class_weight': choice('class_weight', self.class_weight)
            }

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
            'penalty': (
                partial(validate_type, options=list),
                partial(validate_item_in_set, options=(
                    'l1', 'l2', 'elasticnet'))
                ),
            'C': (
                partial(validate_type, options=list),
                partial(validate_length, reference=2, sign='=='),
                partial(validate_subtype, options=(int, float)),
                partial(validate_item, reference=0, sign='>')
                ),
            'tol': (
                partial(validate_type, options=list),
                partial(validate_subtype, options=(int, float)),
                partial(validate_item, reference=0, sign='>')
                ),
            'class_weight': (
                partial(validate_type, options=list),
                partial(validate_item_in_set, options=(None, 'balanced'))
                )
            }

        # Loop over the dictionary items
        for key, value in inputs.items():

            # Loop over the validation functions
            for function in validation_map[key]:

                # Run the validation function
                function(key, value)
