"""Moments objective."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from jax import jit
import jax.numpy as jnp
from numpy import array, concatenate, prod, unravel_index, zeros

# %% Internal package import

from pyanno4rt.datahub import Datahub
from pyanno4rt.optimization.components.objectives import (
    ConventionalObjectiveClass)

# %% Class definition


class Moments(ConventionalObjectiveClass):
    """
    Moments objective class.

    This class provides methods to compute the value and the gradient of the \
    moments objective, as well as to get/set the parameters and the objective \
    weight.

    Parameters
    ----------
    exponents : list
        Exponents.

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
            exponents=None,
            embedding='active',
            weight=1.0,
            link=None,
            identifier=None,
            display=True):

        # Call the superclass constructor to initialize and check attributes
        super().__init__(name='Moments',
                         parameter_name=('exponents',),
                         parameter_category=('vector',),
                         parameter_value=(exponents,),
                         embedding=embedding,
                         weight=weight,
                         link=link,
                         identifier=identifier,
                         display=display)

        # Set the individual parameter value
        self.parameter_value = [float(exponent) for exponent in exponents]

        # Precompile the computation function
        self.jitted_compute = jit(compute, static_argnums=(1, 2))

        # Precompile the differentiation function
        self.jitted_differentiate = jit(differentiate,
                                        static_argnums=(1, 2, 3))

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

        # Initialize the datahub
        hub = Datahub()

        # Concatenate the dose vector(s)
        full_dose = concatenate(args[0])

        # Concatenate the segment index vector(s)
        full_segment_indices = concatenate(tuple(
            hub.segmentation[args[1][i]]['resized_indices']
            for i, _ in enumerate(args[0])))

        # Initialize dose cube
        dose_cube = zeros(hub.dose_information['cube_dimensions'])

        # Insert the dose values into the cube
        dose_cube[unravel_index(full_segment_indices,
                                hub.dose_information['cube_dimensions'],
                                order='F')] = full_dose

        return array(self.jitted_compute(
            dose_cube, tuple(self.parameter_value),
            tuple(hub.dose_information['cube_dimensions']),
            [hub.segmentation[args[1][i]]['resized_indices']
             for i, _ in enumerate(args[0])]))

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

        # Concatenate the dose vector(s)
        full_dose = concatenate(args[0])

        # Concatenate the segment index vector(s)
        full_segment_indices = concatenate(tuple(
            hub.segmentation[args[1][i]]['resized_indices']
            for i, _ in enumerate(args[0])))

        # Initialize dose cube
        dose_cube = zeros(hub.dose_information['cube_dimensions'])

        # Insert the dose values into the cube
        dose_cube[unravel_index(full_segment_indices,
                                hub.dose_information['cube_dimensions'],
                                order='F')] = full_dose

        return array(self.jitted_differentiate(
            dose_cube, tuple(self.parameter_value),
            tuple(hub.dose_information['cube_dimensions']),
            hub.dose_information['number_of_voxels'],
            [hub.segmentation[args[1][i]]['resized_indices']
             for i, _ in enumerate(args[0])])).reshape(
                     (prod(hub.dose_information['cube_dimensions']),))


def compute(
        dose_cube,
        parameter_value,
        cube_dimensions,
        segment_indices):
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

    def compute_scaled_cube(array):
        """Compute the mean of the dose cube scaled by an array."""
        return jnp.sum((array * dose_cube)) / jnp.sum(dose_cube)

    # Get the coefficients of the moment function from the argument
    coeff_1, coeff_2, coeff_3 = tuple(map(jnp.float32, parameter_value))

    # Determine the axis points from a meshed grid
    points_x, points_y, points_z = jnp.meshgrid(
        jnp.array(range(dose_cube.shape[0])),
        jnp.array(range(dose_cube.shape[1])),
        jnp.array(range(dose_cube.shape[2])))

    # Compute the means of the axis points
    mean_x, mean_y, mean_z = tuple(
        map(compute_scaled_cube, (points_x, points_y, points_z)))

    # Compute the moment function numerator
    numerator = jnp.sum((points_x-mean_x)**coeff_1
                        * (points_y-mean_y)**coeff_2
                        * (points_z-mean_z)**coeff_3
                        * dose_cube)

    # Compute the moment function denominator
    denominator = jnp.sum(dose_cube)**((coeff_1+coeff_2+coeff_3)/3 + 1)

    return numerator/denominator


def differentiate(
        dose_cube,
        parameter_value,
        cube_dimensions,
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

    def compute_scaled_cube(array):
        """Compute the mean of the dose cube scaled by an array."""
        return jnp.sum((array * dose_cube)) / jnp.sum(dose_cube)

    # Get the coefficients of the moment function from the argument
    coeff_1, coeff_2, coeff_3 = tuple(map(jnp.float32, parameter_value))

    # Determine the axis points from a meshed grid
    points_x, points_y, points_z = jnp.meshgrid(
        jnp.array(range(dose_cube.shape[0])),
        jnp.array(range(dose_cube.shape[1])),
        jnp.array(range(dose_cube.shape[2])))

    # Compute the means of the axis points
    mean_x, mean_y, mean_z = tuple(
        map(compute_scaled_cube, (points_x, points_y, points_z)))

    # Compute derivative mean
    dx_bar_dD = (points_x-mean_x) / jnp.sum(dose_cube)
    dy_bar_dD = (points_y-mean_y) / jnp.sum(dose_cube)
    dz_bar_dD = (points_z-mean_z) / jnp.sum(dose_cube)

    # Calculate derivative u
    du_dD = (- coeff_1 * jnp.sum((points_x-mean_x)**(coeff_1 - 1)
                                 * (points_y-mean_y)**coeff_2
                                 * (points_z-mean_z)**coeff_3
                                 * dose_cube) * dx_bar_dD
             - coeff_1 * jnp.sum((points_x-mean_x)**coeff_1
                                 * (points_y-mean_y)**(coeff_2 - 1)
                                 * (points_z-mean_z)**coeff_3
                                 * dose_cube) * dy_bar_dD
             - coeff_3 * jnp.sum((points_x-mean_x)**coeff_1
                                 * (points_y-mean_y)**coeff_2
                                 * (points_z-mean_z)**(coeff_3 - 1)
                                 * dose_cube) * dz_bar_dD
             + (points_x-mean_x)**coeff_1
             * (points_y-mean_y)**coeff_2
             * (points_z-mean_z)**coeff_3)

    # Calculate derivative v
    dv_dD = ((jnp.sum(jnp.array(parameter_value))/3 + 1)
             * jnp.sum(dose_cube)**jnp.sum(jnp.array(parameter_value))/3)

    # Calculate full derivative
    u = jnp.sum((points_x-mean_x)**coeff_1
                * (points_y-mean_y)**coeff_2
                * (points_z-mean_z)**coeff_3
                * dose_cube)

    # 
    v = jnp.sum(dose_cube)**(jnp.sum(jnp.array(parameter_value)) / 3 + 1)

    return (du_dD*v - u*dv_dD) / v**2
