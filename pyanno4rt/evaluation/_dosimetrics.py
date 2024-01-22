"""Dosimetrics computation."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from numpy import floor, linspace, power, sort, unravel_index
from scipy.interpolate import interp1d
from statistics import mean

# %% Internal package import

from pyanno4rt.datahub import Datahub

# %% Class definition


class Dosimetrics():
    """
    Dosimetrics computation class.

    This class provides methods to compute dosimetrics as a means to evaluate \
    dose distributions from treatment plans across all segments. \
    Dosimetrics include statistical location and dispersion measures, DVH \
    indicators as well as conformity (CI) and homogeneity indices (HI).

    Parameters
    ----------
    reference_volume : list
        Reference volumes for which to calculate the inverse DVH values.

    reference_dose : list
        Reference dose values for which to calculate the DVH values.

    display_segments : list
        Names of the segmented structures to be displayed.

    display_metrics : list
        Names of the metrics to be displayed.

    Attributes
    ----------
    reference_volume : list
        See 'Parameters'.

    reference_dose : list
        See 'Parameters'.

    display_segments : list
        See 'Parameters'.

    display_metrics : list
        See 'Parameters'.
    """

    def __init__(
            self,
            reference_volume,
            reference_dose,
            display_segments,
            display_metrics):

        # Initialize the datahub
        hub = Datahub()

        # Log a message about the initialization of the class
        hub.logger.display_info("Initializing dosimetrics class ...")

        # Get the sorted reference volumes and doses from the arguments
        self.reference_volume = sorted(reference_volume)
        self.reference_dose = sorted(reference_dose)

        # Get the (default) segments and metrics to display from the arguments
        self.display_segments = (display_segments if len(display_segments) > 0
                                 else list(hub.segmentation))
        self.display_metrics = (display_metrics if len(display_metrics) > 0
                                else ['mean', 'std', 'max', 'min', 'Dx', 'Vx',
                                      'CI', 'HI'])

    def compute(
            self,
            dose_cube):
        """
        Compute the dosimetrics for all segments.

        Parameters
        ----------
        dose_cube : ndarray
            Three-dimensional array with the dose values (CT resolution).
        """

        # Initialize the datahub
        hub = Datahub()

        # Log a message about the dosimetrics computation
        hub.logger.display_info("Computing dosimetrics for all segments ...")

        # Check if the reference dose list is empty
        if len(self.reference_dose) == 0:

            # Apply default reference dose values
            self.reference_dose = list(
                floor(linspace(0, dose_cube.max(), 5)*10)/10)

        # Initialize the dosimetrics dictionary
        dosimetrics = {segment: {} for segment in hub.segmentation}

        # Loop over the segments
        for segment in dosimetrics:

            # Get the sorted dose vector
            dose = sort(dose_cube[unravel_index(
                hub.segmentation[segment]['raw_indices'],
                hub.computed_tomography['cube_dimensions'], order='F')])

            # Get the length of the dose vector
            dose_length = len(dose)

            # Check if any dose values are present
            if dose_length > 0:

                # Initialize the linear dose interpolator
                interpolator = interp1d(linspace(0, 1, dose_length), dose,
                                        'linear', fill_value='extrapolate')

                # Compute the base statistics from the dose vector
                dosimetrics[segment].update({
                    metric: getattr(dose, metric)()
                    for metric in ('mean', 'std', 'min', 'max')})

                # Compute the dose quantiles from the reference volumes
                dosimetrics[segment].update(
                    {'_'.join(('Dx', str(value))): interpolator(1-value/100)
                     for value in self.reference_volume})

                # Compute the relative volumes from the reference doses
                dosimetrics[segment].update(
                    {'_'.join(('Vx', str(value))):
                     ((dose >= value).sum()/dose_length)
                     for value in self.reference_dose})

                # Check if the segment is a target volume with objective
                if (hub.segmentation[segment]['type'].lower() == 'target'
                        and hub.segmentation[segment]['objective']):

                    # Initialize the target dose
                    target_dose = None

                    # Get the target objective
                    target_objective = hub.segmentation[segment]['objective']

                    # Set the relevant objective names
                    names = ('Squared Deviation', 'Squared Underdosing')

                    # Check if the objective is a tuple with relevant elements
                    if (isinstance(target_objective, tuple)
                            and any(objective.name in names
                                    for objective in target_objective)):

                        # Compute the target dose
                        target_dose = mean(
                            objective.get_parameter_value()[0]
                            for objective in target_objective
                            if objective.name in names)

                    # Else, check if the objective is not a tuple but relevant
                    elif (not isinstance(target_objective, tuple)
                          and target_objective.name in names):

                        # Set the target dose to the parameter value
                        target_dose = target_objective.get_parameter_value()[0]

                    # Check if the target dose has a value
                    if target_dose:

                        # Set the dose threshold
                        threshold = 0.95*target_dose

                        # Get the rounded threshold as a string
                        sub = str(round(target_dose*100)/100)

                        # Add the CI values to the dosimetrics dictionary
                        dosimetrics[segment][''.join(('CI_', sub, 'Gy'))] = (
                            power((dose >= threshold).sum(), 2)
                            / (dose_length*(dose_cube >= threshold).sum()))

                        # Add the HI values to the dosimetrics dictionary
                        dosimetrics[segment][''.join(('HI_', sub, 'Gy'))] = (
                            ((interpolator(0.95)-interpolator(0.05))
                             / target_dose) * 100)

        # Add the segment and metric names to be displayed
        dosimetrics['display_segments'] = self.display_segments
        dosimetrics['display_metrics'] = self.display_metrics

        # Enter the dosimetrics dictionary into the datahub
        hub.dosimetrics = dosimetrics
