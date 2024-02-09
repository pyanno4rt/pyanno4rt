"""Dose uniformity objective."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from math import sqrt
from numba import njit
from numpy import concatenate, zeros

# %% Internal package import

from pyanno4rt.datahub import Datahub
from pyanno4rt.optimization.components.objectives import (
    ConventionalObjectiveClass)

# %% Class definition


class DoseUniformity(ConventionalObjectiveClass):
    """
    Dose uniformity objective class.

    This class provides methods to compute the value and the gradient of the \
    dose uniformity objective, as well as to get/set the parameters and the \
    objective weight.

    Parameters
    ----------
    embedding : {'active', 'passive'}, default = 'active'
        Mode of embedding for the objective. In 'passive' mode, the objective \
        value is computed and tracked, but not included in the optimization \
        problem. In 'active' mode, however, both objective value and \
        gradient vector are computed and included in the optimization problem.

    weight : int or float, default = 1.0
        Weight of the objective function.

    link : list, default = None
        Link to additional segments for joint evaluation.

    Attributes
    ----------
    name : string
        Name of the objective class.

    parameter_name : tuple
        Name of the objective parameters.

    parameter_category : tuple
        Category of the objective parameters.

    embedding : {'active', 'passive'}
        See 'Parameters'.

    weight : float
        See 'Parameters'.

    link : list
        See 'Parameters'.

    adjusted_parameters : bool
        Indicator for the adjustment of the dose-related parameters.

    DEPENDS_ON_MODEL : bool
        Indicator for the model dependency of the objective.

    parameter_value : list
        Value of the objective parameters.
    """

    def __init__(
            self,
            embedding='active',
            weight=1.0,
            link=None,
            identifier=None,
            display=True):

        # Call the superclass constructor to initialize and check attributes
        super().__init__(name='Dose Uniformity',
                         parameter_name=(),
                         parameter_category=(),
                         parameter_value=(),
                         embedding=embedding,
                         weight=weight,
                         link=link,
                         identifier=identifier,
                         display=display)

        # Set the individual parameter value
        self.parameter_value = []

    def compute_objective_value(
            self,
            *args):
        """
        Call the computation function for the objective value.

        Parameters
        ----------
        args : tuple
            Keyworded parameters. args[0] should give the dose vector(s) to \
            evaluate.

        Returns
        -------
        float
            Value of the objective function.
        """
        return compute(args[0])

    def compute_gradient_value(
            self,
            *args):
        """
        Call the computation function for the gradient vector.

        Parameters
        ----------
        args : tuple
            Keyworded parameters. args[0] should give the dose vector(s) to \
            evaluate, args[1] the corresponding segment(s).

        Returns
        -------
        ndarray
            Value of the gradient vector.
        """
        # Initialize the datahub
        hub = Datahub()

        # Get the segment indices
        segment_indices = tuple(hub.segmentation[args[1][i]]['resized_indices']
                                for i, _ in enumerate(args[0]))

        return differentiate(args[0], hub.dose_information['number_of_voxels'],
                             segment_indices)


@njit
def compute(
        dose):
    """
    Compute the value of the objective.

    Parameters
    ----------
    dose : tuple
        Values of the dose for a single or multiple segments.

    Returns
    -------
    float
        Value of the objective function.

    Notes
    -----
    This computation function has been outsourced to make it jittable. Called \
    by ``compute_objective_value(*args)``.
    """

    # Concatenate the dose arrays
    full_dose = concatenate(dose)

    return full_dose.std()*sqrt(len(full_dose)/(len(full_dose)-1))


@njit
def differentiate(
        dose,
        number_of_voxels,
        segment_indices):
    """
    Compute the value of the gradient.

    Parameters
    ----------
    dose : tuple
        Value of the dose for the segment(s).

    parameter_value : list
        Value of the objective parameters.

    number_of_voxels : int
        Total number of dose voxels.

    segment_indices : list
        Indices of the segment(s).

    Returns
    -------
    ndarray
        Gradient vector of the objective function.

    Notes
    -----
    This differentiation function has been outsourced to make it jittable. \
    Called by ``compute_gradient_value(*args)``.
    """

    # Initialize the objective gradient
    objective_gradient = zeros((number_of_voxels,))

    # Concatenate the dose arrays
    full_dose = concatenate(dose)

    # Concatenate the segment index arrays
    full_indices = concatenate(segment_indices)

    # Compute the segment-wise weighted subgradient
    gradient = (full_dose - full_dose.mean()) / (
        (len(full_dose)-1) * full_dose.std())

    # Add the subgradients to the objective gradient
    objective_gradient[full_indices] = gradient.reshape(-1)

    return objective_gradient
