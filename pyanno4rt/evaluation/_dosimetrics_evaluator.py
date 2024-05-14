"""Dosimetrics evaluation."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from statistics import mean

from numpy import floor, linspace, power, sort, unravel_index
from scipy.interpolate import interp1d

# %% Internal package import

from pyanno4rt.datahub import Datahub

# %% Class definition


class DosimetricsEvaluator():
    """
    Dosimetrics evaluation class.

    This class provides methods to evaluate dosimetrics as a means to \
    quantify dose distributions from a treatment plan across the segments. \
    Dosimetrics include statistical location and dispersion measures, DVH \
    indicators as well as conformity (CI) and homogeneity index (HI).

    Parameters
    ----------
    reference_volume : list
        Reference volumes for which to evaluate the inverse DVH indicators.

    reference_dose : list
        Reference dose values for which to evaluate the DVH indicators.

    display_segments : list
        Names of the segmented structures to be displayed.

    display_metrics : list
        Names of the metrics to be displayed.

    Attributes
    ----------
    reference_volume : tuple
        See 'Parameters'.

    reference_dose : tuple
        See 'Parameters'.

    display_segments : tuple
        See 'Parameters'.

    display_metrics : tuple
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
        hub.logger.display_info("Initializing dosimetrics evaluator ...")

        # Get the sorted reference volumes and doses from the arguments
        self.reference_volume, self.reference_dose = map(
            sorted, (tuple(reference_volume), tuple(reference_dose)))

        # Check if the length of the "display_segments" argument is zero
        if len(display_segments) == 0:

            # Get the display segments from the datahub
            self.display_segments = tuple(hub.segmentation)

        else:

            # Get the display segments from the argument
            self.display_segments = tuple(display_segments)

        # Check if the length of the "display_metrics" argument is zero
        if len(display_metrics) == 0:

            # Get the default display metrics
            self.display_metrics = (
                'mean', 'std', 'max', 'min', 'Dx', 'Vx', 'CI', 'HI')

        else:

            # Get the display metrics from the argument
            self.display_metrics = tuple(display_metrics)

    def evaluate(
            self,
            dose_cube):
        """
        Evaluate the dosimetrics for all segments.

        Parameters
        ----------
        dose_cube : ndarray
            Three-dimensional array with the dose values (CT resolution).
        """

        # Initialize the datahub
        hub = Datahub()

        # Log a message about the dosimetrics evaluation
        hub.logger.display_info("Evaluating dosimetrics for all segments ...")

        # Check if the reference dose list is empty
        if len(self.reference_dose) == 0:

            # Apply default reference dose values
            self.reference_dose = tuple(
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
                dosimetrics[segment] |= {
                    metric: getattr(dose, metric)()
                    for metric in ('mean', 'std', 'min', 'max')}

                # Compute the dose quantiles from the reference volumes
                dosimetrics[segment] |= {
                    f'Dx_{value}': interpolator(1-value/100)
                    for value in self.reference_volume}

                # Compute the relative volumes from the reference doses
                dosimetrics[segment] |= {
                    f'Vx_{value}': ((dose >= value).sum()/dose_length)
                    for value in self.reference_dose}

                # Check if the segment is a target volume with objective
                if (hub.segmentation[segment]['type'].lower() in (
                        'tv', 'target', 'boost', 'tumor')
                        and hub.segmentation[segment]['objective']):

                    # Get the target objective
                    target_objective = hub.segmentation[segment]['objective']

                    # Set the relevant objective names
                    names = ('Squared Deviation', 'Squared Underdosing')

                    # Check if the objective is a tuple with relevant elements
                    if (isinstance(target_objective, list)
                            and any(objective.name in names
                                    for objective in target_objective)):

                        # Set the target dose to the mean parameter value
                        target_dose = mean(
                            objective.get_parameter_value()[0]
                            for objective in target_objective
                            if objective.name in names)

                    # Else, check if the objective is not a tuple but relevant
                    elif (not isinstance(target_objective, list)
                          and target_objective.name in names):

                        # Set the target dose to the parameter value
                        target_dose = target_objective.get_parameter_value()[0]

                    else:

                        # Set the target dose to None
                        target_dose = None

                    # Check if the target dose has a value
                    if target_dose:

                        # Set the dose threshold
                        threshold = 0.95*target_dose

                        # Get the rounded threshold as a string
                        sub = str(round(target_dose*100)/100)

                        # Add the conformity index to the dosimetrics
                        dosimetrics[segment][f'CI_{sub}Gy'] = (
                            power((dose >= threshold).sum(), 2)
                            / (dose_length*(dose_cube >= threshold).sum()))

                        # Add the homogeneity index to the dosimetrics
                        dosimetrics[segment][f'HI_{sub}Gy'] = (
                            ((interpolator(0.95)-interpolator(0.05))
                             / target_dose) * 100)

        # Add the segment and metric names to be displayed
        dosimetrics['display_segments'] = self.display_segments
        dosimetrics['display_metrics'] = self.display_metrics

        # Enter the dosimetrics dictionary into the datahub
        hub.dosimetrics = dosimetrics
