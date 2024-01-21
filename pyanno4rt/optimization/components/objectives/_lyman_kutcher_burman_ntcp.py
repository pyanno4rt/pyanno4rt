"""Lyman-Kutcher-Burman (LKB) NTCP objective."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from math import erf, pi, sqrt
from numba import njit
from numpy import concatenate, exp, power, zeros

# %% Internal package import

from pyanno4rt.datahub import Datahub
from pyanno4rt.optimization.components.objectives import (
    RadiobiologyObjectiveClass)

# %% Class definition


class LymanKutcherBurmanNTCP(RadiobiologyObjectiveClass):
    """
    Lyman-Kutcher-Burman (LKB) NTCP objective class.

    This class provides methods to compute the value and the gradient of the \
    LKB NTCP objective, as well as to get/set the parameters and the \
    objective weight.

    Parameters
    ----------
    tolerance_dose_50 : ...
        ...

    slope_parameter : ...
        ...

    volume_parameter : ...
        ...

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

    __dose__ : ...
        ...

    __gmd__ : ...
        ...
    """

    def __init__(
            self,
            tolerance_dose_50=None,
            slope_parameter=None,
            volume_parameter=None,
            embedding='active',
            weight=1.0,
            link=None,
            identifier=None,
            display=True):

        # Call the superclass constructor to initialize and check attributes
        super().__init__(name='Lyman-Kutcher-Burman NTCP',
                         parameter_name=('tolerance_dose_50',
                                         'slope_parameter',
                                         'volume_parameter'),
                         parameter_category=('dose', 'coefficient',
                                             'coefficient'),
                         parameter_value=(tolerance_dose_50,
                                          slope_parameter,
                                          volume_parameter),
                         embedding=embedding,
                         weight=weight,
                         link=link,
                         identifier=identifier,
                         display=display)

        # Set the individual parameter value
        self.parameter_value = [float(tolerance_dose_50),
                                float(slope_parameter),
                                float(volume_parameter)]

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
        return compute(args[0], self.parameter_value)

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

        return differentiate(args[0], self.parameter_value,
                             hub.dose_information['number_of_voxels'],
                             segment_indices)


@njit
def compute(dose, parameter_value):
    """
    Compute the value of the objective.

    Parameters
    ----------
    dose : tuple
        Value of the dose for the segment(s).

    parameter_value : list
        Values of the objective parameters.

    Returns
    -------
    float
        Value of the objective function.

    Notes
    -----
    This computation function has been outsourced to make it jittable. Called \
    by ``compute_objective_value(*args)``.
    """

    # Concatenate the dose vector(s)
    full_dose = concatenate(dose)

    # Compute the generalized mean dose
    gEUD = power((1/len(full_dose))*sum(power(full_dose, parameter_value[2])),
                 parameter_value[2])

    # Compute the objective value with the error function
    objective_value = 0.5*(1+erf((gEUD-parameter_value[0]) /
                                 (sqrt(2) * parameter_value[1]
                                  * parameter_value[0])))

    return objective_value


@njit
def differentiate(dose, parameter_value, number_of_voxels, segment_indices):
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

    # Compute the generalized mean dose
    gEUD = power((1/len(full_dose))*sum(power(full_dose, parameter_value[2])),
                 parameter_value[2])

    # Compute the dose gradient of the gEUD
    gEUD_gradient = (1/power(len(full_dose), parameter_value[2])
                     * power(sum(power(full_dose, 1/parameter_value[2])),
                             parameter_value[2]-1)
                     * power(full_dose, 1/parameter_value[2] - 1))

    # Compute the gEUD gradient of the objective
    ntcp_gradient = (2*exp(-power((gEUD-parameter_value[0]) /
                                  (sqrt(2) * parameter_value[1]
                                   * parameter_value[0]), 2))
                     / sqrt(pi))

    # Compute the full objective gradient
    total_gradient = gEUD_gradient * ntcp_gradient

    # Add the subgradients to the objective gradient
    objective_gradient[full_indices] = total_gradient.reshape(-1)

    return objective_gradient
