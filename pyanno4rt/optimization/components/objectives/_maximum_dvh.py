"""Maximum DVH objective."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from numba import njit
from numpy import dot, logical_or, quantile, sort, zeros

# %% Internal package import

from pyanno4rt.datahub import Datahub
from pyanno4rt.optimization.components.objectives import ObjectiveClass

# %% Class definition


class MaximumDVH(ObjectiveClass):
    """
    Maximum dose-volume histogram (Maximum DVH) objective class.

    This class provides methods to compute the value and the gradient of the \
    maximum DVH objective, as well as to get/set the parameters and the \
    objective weight.

    Parameters
    ----------
    target_dose : int or float, default = 30.0
        Reference value for the dose.

    maximum_volume : int or float, default = 95.0
        Maximum volume level for computing the nearest dose quantile.

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
            target_dose=30.0,
            maximum_volume=95.0,
            embedding='active',
            weight=1.0,
            link=None,
            identifier=None,
            display=True):

        # Call the superclass constructor to initialize and check attributes
        super().__init__(name='Maximum DVH',
                         parameter_name=('target_dose', 'maximum_volume'),
                         parameter_category=('dose', 'volume'),
                         embedding=embedding,
                         weight=weight,
                         link=link,
                         identifier=identifier,
                         display=display)

        # Set the individual parameter value
        self.parameter_value = [float(target_dose),
                                float(maximum_volume)/100]

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
        # Get the parameter term
        maximum_volume = self.parameter_value[1]

        # Compute the dose quantile
        dose_quantile = [quantile(
            dose_sorted, maximum_volume, interpolation='lower')
            for dose_sorted in [sort(dose)[::-1] for dose in args[0]]]

        return compute(args[0], self.parameter_value, dose_quantile)

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

        # Get the parameter term
        max_volume = self.parameter_value[1]

        # Compute the dose quantile
        dose_quantile = [quantile(
            dose_sorted, max_volume, interpolation='lower')
            for dose_sorted in [sort(dose)[::-1] for dose in args[0]]]

        return differentiate(args[0], self.parameter_value, dose_quantile,
                             hub.dose_information['number_of_voxels'],
                             [hub.segmentation[args[1][i]]['resized_indices']
                              for i, _ in enumerate(args[0])])


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
    # Get the parameter term
    reference_dose = parameter_value[0]

    # Center the dose values with the reference dose
    deviation = [dos - reference_dose for dos in dose]

    # Generate a boolean mask to filter dose values
    condition = [logical_or(dose[i] < reference_dose,
                            dose[i] > dose_quantile[i])
                 for i, _ in enumerate(dose)]

    # Set all elements to zero for which the filter holds true
    for i, _ in enumerate(dose):
        deviation[i][condition[i]] = 0

    return (sum([(1/len(dose[i])) * dot(deviation[i], deviation[i])
                 for i, _ in enumerate(dose)]))


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
    # Get the parameter term
    reference_dose = parameter_value[0]

    # Center the dose values with the reference dose
    deviation = [dos - reference_dose for dos in dose]

    # Generate a boolean mask to filter dose values
    condition = [logical_or(dose[i] < reference_dose,
                            dose[i] > dose_quantile[i])
                 for i, _ in enumerate(dose)]

    # Set all elements to zero for which the filter holds true
    for i, _ in enumerate(dose):
        deviation[i][condition[i]] = 0

    # Initialize the objective gradient
    objective_gradient = zeros((number_of_voxels,))

    # Compute the segment-wise subgradient
    gradient = [(2/len(dose[i])) * deviation[i] for i, _ in enumerate(dose)]

    # Add the subgradients to the objective gradient
    for i, _ in enumerate(segment_indices):
        objective_gradient[segment_indices[i]] = gradient[i].reshape(-1)

    return objective_gradient
