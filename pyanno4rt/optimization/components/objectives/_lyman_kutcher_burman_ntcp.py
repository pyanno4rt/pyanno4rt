"""Lyman-Kutcher-Burman (LKB) NTCP objective."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from numpy import dot, zeros

# %% Internal package import

from pyanno4rt.datahub import Datahub
from pyanno4rt.optimization.components.objectives._objective_class import (
    ObjectiveClass)

# %% Class definition


class LymanKutcherBurmanNTCP(ObjectiveClass):
    """Lyman-Kutcher-Burman (LKB) NTCP objective class."""

    tissue_parameters = {'PAROTID_LT': [],
                         'PAROTID_RT': []}

    name = 'Lyman-Kutcher-Burman NTCP'
    parameter_name = ['reference_volume', 'TD50', 'volume_parameter',
                      'slope_parameter']
    parameter_category = ['volume', 'dose', 'parameter', 'parameter']
    parameter_value = [1, 60, 1, 1]
    weight = 1.0

    def __init__(
            self,
            reference_dose=parameter_value[0],
            model=parameter_value[1],
            diffType=parameter_value[2],
            weight=1.0,
            link=None):

        self.adjusted_params = False
        self.link = link

        self.parameter_value = [reference_dose
                                if isinstance(reference_dose, float)
                                else float(reference_dose),
                                model
                                if isinstance(model, float)
                                else float(model),
                                diffType
                                if isinstance(diffType, float)
                                else float(diffType)]
        self.weight = weight if isinstance(weight, float) else float(weight)

        super(LymanKutcherBurmanNTCP, LymanKutcherBurmanNTCP).check_objective(
            self.name, self.parameter_name, self.parameter_category,
            self.parameter_value, self.weight)

    def compute_objective_value(
            self,
            *args):
        """Compute the value of the objective function."""
        objective_value = 0

        for i in range(0, len(args[0])):

            deviation = args[0][i] - self.parameter_value
            objective_value += 1 / len(args[0][i]) * dot(deviation.T,
                                                         deviation)

        return objective_value

    def compute_gradient_value(
            self,
            *args):
        """Compute the value of the gradient."""
        hub = Datahub()
        objective_gradient = zeros((hub.dose_information['number_of_voxels'],))

        for i in range(0, len(args[0])):

            deviation = args[0][i] - self.parameter_value
            gradient = 2 / len(args[0][i]) * deviation

            objective_gradient[
                hub.segmentation[args[1][i]]['resized_indices']] = gradient

        return objective_gradient

    def get_parameter_value(self):
        """Get the value of the parameter."""
        return self.parameter_value

    def set_parameter_value(
            self,
            *args):
        """Set the value of the parameter."""
        self.parameter_value = args[0]

    def get_weight_value(self):
        """Get the value of the weight."""
        return self.weight

    def set_weight_value(
            self,
            *args):
        """Set the value of the weight."""
        self.weight = args[0]
