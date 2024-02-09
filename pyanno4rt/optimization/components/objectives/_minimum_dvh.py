"""Minimum DVH objective."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from numba import njit
from numpy import concatenate, logical_or, quantile, sort, zeros

# %% Internal package import

from pyanno4rt.datahub import Datahub
from pyanno4rt.optimization.components.objectives import (
    ConventionalObjectiveClass)

# %% Class definition


class MinimumDVH(ConventionalObjectiveClass):
    """
    Minimum dose-volume histogram (Minimum DVH) objective class.

    This class provides methods to compute the value and the gradient of the \
    minimum DVH objective, as well as to get/set the parameters and the \
    objective weight.

    Parameters
    ----------
    target_dose : int or float
        Minimum value for the dose.

    minimum_volume : int or float
        Minimum volume level for computing the nearest dose quantile.

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

    parameter_name : list
        Name of the objective parameters.

    parameter_category : list
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
            target_dose=None,
            minimum_volume=None,
            embedding='active',
            weight=1.0,
            link=None,
            identifier=None,
            display=True):

        # Call the superclass constructor to initialize and check attributes
        super().__init__(name='Minimum DVH',
                         parameter_name=('target_dose', 'minimum_volume'),
                         parameter_category=('dose', 'volume'),
                         parameter_value=(target_dose, minimum_volume),
                         embedding=embedding,
                         weight=weight,
                         link=link,
                         identifier=identifier,
                         display=display)

        # Set the individual parameter value
        self.parameter_value = [float(target_dose),
                                float(minimum_volume)/100]

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

        # Concatenate the dose arrays
        full_dose = concatenate(args[0])

        # Compute the dose quantile
        dose_quantile = quantile(sort(full_dose)[::-1],
                                 self.parameter_value[1],
                                 interpolation='lower')

        return compute(full_dose, self.parameter_value, dose_quantile)

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

        # Concatenate the dose arrays
        full_dose = concatenate(args[0])

        # Compute the dose quantile
        dose_quantile = quantile(sort(full_dose)[::-1],
                                 self.parameter_value[1],
                                 interpolation='lower')

        # Get the segment indices
        segment_indices = tuple(hub.segmentation[args[1][i]]['resized_indices']
                                for i, _ in enumerate(args[0]))

        return differentiate(full_dose, self.parameter_value, dose_quantile,
                             hub.dose_information['number_of_voxels'],
                             segment_indices)


@njit
def compute(
        dose,
        parameter_value,
        dose_quantile):
    """
    Compute the value of the objective.

    Parameters
    ----------
    dose : tuple
        Value of the dose for the segment(s).

    parameter_value : list
        Values of the objective parameters.

    dose_quantile : list
        Value of the dose quantile for the segment(s).

    Returns
    -------
    float
        Value of the objective function.

    Notes
    -----
    This computation function has been outsourced to make it jittable. Called \
    by ``compute_objective_value(*args)``.
    """

    # Center the dose values with the reference dose
    deviation = dose - parameter_value[0]

    # Generate a boolean mask to filter dose values
    condition = logical_or(dose > parameter_value[0], dose < dose_quantile)

    # Set all elements to zero for which the filter holds true
    deviation[condition] = 0

    return (1/len(dose)) * (deviation @ deviation)


@njit
def differentiate(
        dose,
        parameter_value,
        dose_quantile,
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

    dose_quantile : list
        Value of the dose quantile for the segment(s).

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

    # Concatenate the segment index arrays
    full_indices = concatenate(segment_indices)

    # Get the parameter term
    reference_dose = parameter_value[0]

    # Center the dose values with the reference dose
    deviation = dose - reference_dose

    # Generate a boolean mask to filter dose values
    condition = logical_or(dose > reference_dose, dose < dose_quantile)

    # Set all elements to zero for which the filter holds true
    deviation[condition] = 0

    # Compute the gradient
    objective_gradient[full_indices] = (2*deviation/len(dose)).reshape(-1)

    return objective_gradient
