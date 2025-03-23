"""Base treatment plan."""

# Author: Tim Ortkamp

# %% Internal package import

# Functional classes
from pyanno4rt.base import Configuration, Evaluation, Optimization
from pyanno4rt.logging import Logger
from pyanno4rt.datahub import Datahub

# Treatment plan configuration
from pyanno4rt.patient import PatientLoader
from pyanno4rt.plan import PlanGenerator
from pyanno4rt.dose_info import DoseInfoGenerator

# Treatment plan optimization
from pyanno4rt.optimization import FluenceOptimizer

# Treatment plan evaluation
from pyanno4rt.evaluation import DVHEvaluator
from pyanno4rt.evaluation import DosimetricsEvaluator

# Treatment plan visualization
from pyanno4rt.visualization import Visualizer

# Supporting functions
from pyanno4rt.checking import check_type
from pyanno4rt.tools import (
    apply, get_machine_learning_constraints, get_machine_learning_objectives)

# %% Class definition


class TreatmentPlan():
    """
    Base treatment plan class.

    This class enables configuration, optimization, evaluation, and \
    visualization of individual IMRT treatment plans.

    Parameters
    ----------
    configuration : object of class \
        :class:`~pyanno4rt.base._configuration.Configuration`
        The object used to handle the plan configuration parameters.

    optimization : object of class \
        :class:`~pyanno4rt.base._optimization.Optimization`
        The object used to handle the plan optimization parameters.

    evaluation : object of class \
        :class:`~pyanno4rt.base._evaluation.Evaluation`
        The object used to handle the plan evaluation parameters.

    Attributes
    ----------
    configuration : object of class \
        :class:`~pyanno4rt.base._configuration.Configuration`
        See 'Parameters'.

    optimization : object of class \
        :class:`~pyanno4rt.base._optimization.Optimization`
        See 'Parameters'.

    evaluation : object of class \
        :class:`~pyanno4rt.base._evaluation.Evaluation`
        See 'Parameters'.

    logger : None or object of class \
        :class:`~pyanno4rt.logging._logger.Logger`
        The object used to print and store logging messages.

    datahub : None or object of class \
        :class:`~pyanno4rt.datahub._datahub.Datahub`
        The object used to manage and distribute information units.

    patient_loader : None or object of class \
        :class:`~pyanno4rt.patient._patient_loader.PatientLoader`
        The object used to import and type-convert CT and segmentation data.

    plan_generator : None or object of class \
        :class:`~pyanno4rt.plan._plan_generator.PlanGenerator`
        The object used to set and type-convert plan properties.

    dose_info_generator : None or object of class \
        :class:`~pyanno4rt.dose_info._dose_info_generator.DoseInfoGenerator`
        The object used to specify and type-convert dose (grid) properties.

    fluence_optimizer : None or object of class \
        :class:`~pyanno4rt.optimization._fluence_optimizer.FluenceOptimizer`
        The object used to solve the fluence optimization problem.

    dose_histogram : None or object of class \
        :class:`~pyanno4rt.evaluation._dvh.DVHEvaluator`
        The object used to evaluate the dose-volume histogram (DVH).

    dosimetrics : None or object of class \
        :class:`~pyanno4rt.evaluation._dosimetrics.DosimetricsEvaluator`
        The object used to evaluate the dosimetrics.

    visualizer : None or object of class \
        :class:`~pyanno4rt.visualization._visualizer.Visualizer`
        The object used to visualize the treatment plan.

    Example
    -------
    Our Read the Docs page (https://pyanno4rt.readthedocs.io/en/latest/) \
    features a step-by-step example for the application of this class. You \
    will also find code templates there, e.g. for the optimization components.
    """

    def __init__(
            self,
            configuration,
            optimization,
            evaluation):

        # Check the input arguments
        check_type('configuration', configuration, Configuration)
        check_type('optimization', optimization, Optimization)
        check_type('evaluation', evaluation, Evaluation)

        # Initialize the plan parameter attributes
        self.configuration = configuration
        self.optimization = optimization
        self.evaluation = evaluation

        # Initialize the instance attributes
        self.logger = Logger(
            self.configuration.label, self.configuration.min_log_level)
        self.datahub = Datahub(self.configuration.label, self.logger)
        self.patient_loader = None
        self.plan_generator = None
        self.dose_info_generator = None
        self.fluence_optimizer = None
        self.dose_histogram = None
        self.dosimetrics = None
        self.visualizer = None

    def configure(self):
        """Initialize the configuration classes and process the input data."""

        # Reset the treatment plan label in the datahub
        Datahub.label = self.configuration.label

        # Initialize the patient loader
        self.patient_loader = PatientLoader(
            imaging_path=self.configuration.imaging_path)

        # Load the patient data
        self.patient_loader.load()

        # Initialize the plan generator
        self.plan_generator = PlanGenerator(
            modality=self.configuration.modality,
            components=self.optimization.components)

        # Generate the plan information
        self.plan_generator.generate()

        # Initialize the dose information generator
        self.dose_info_generator = DoseInfoGenerator(
            number_of_fractions=self.configuration.number_of_fractions,
            dose_matrix_path=self.configuration.dose_matrix_path,
            dose_resolution=self.configuration.dose_resolution)

        # Generate the dose information
        self.dose_info_generator.generate()

        # Increment the state
        self.datahub.state = 1

    def model(self):
        """Add the machine learning outcome models to the components."""

        # Reset the treatment plan label in the datahub
        Datahub.label = self.configuration.label

        # Check if the plan has not been configured yet
        if any(getattr(self, attribute) is None for attribute in (
                'logger', 'datahub', 'patient_loader', 'plan_generator',
                'dose_info_generator')):

            # Log a message about the non-configured plan
            self.logger.display_error(
                "Please configure the treatment plan before modeling!")

        else:

            # Get the segmentation dictionary
            segmentation = Datahub().segmentation

            # Add the machine learning outcome models to the components
            apply(lambda component: component.add_model(), (
                get_machine_learning_constraints(segmentation)
                + get_machine_learning_objectives(segmentation)))

            # Increment the state
            self.datahub.state = 2

    def optimize(self):
        """Initialize the optimization classes and solve the problem."""

        # Reset the treatment plan label in the datahub
        Datahub.label = self.configuration.label

        # Get the segmentation dictionary
        segmentation = Datahub().segmentation

        # Check if the plan has not been configured yet
        if any(getattr(self, attribute) is None for attribute in (
                'logger', 'datahub', 'patient_loader', 'plan_generator',
                'dose_info_generator')):

            # Log a message about the non-configured plan
            self.logger.display_error(
                "Please configure the treatment plan before optimization!")

        # Check if machine learning components have not been modeled
        elif any(getattr(component, 'model') is None for component in (
                get_machine_learning_constraints(segmentation)
                + get_machine_learning_objectives(segmentation))):

            # Log a message about the non-modeled components
            self.logger.display_error(
                "Please add the machine learning models before optimization!")

        else:

            # Initialize the fluence optimizer
            self.fluence_optimizer = FluenceOptimizer(
                method=self.optimization.method,
                solver=self.optimization.solver,
                algorithm=self.optimization.algorithm,
                initial_strategy=self.optimization.initial_strategy,
                initial_fluence_vector=(
                    self.optimization.initial_fluence_vector),
                lower_variable_bounds=self.optimization.lower_variable_bounds,
                upper_variable_bounds=self.optimization.upper_variable_bounds,
                maximum_iterations=self.optimization.maximum_iterations,
                tolerance=self.optimization.tolerance)

            # Solve the optimization problem
            self.fluence_optimizer.solve()

            # Increment the state
            self.datahub.state = 3

    def evaluate(self):
        """Initialize the evaluation classes and compute the plan metrics."""

        # Reset the treatment plan label in the datahub
        Datahub.label = self.configuration.label

        # Check if the plan has not been optimized yet
        if (getattr(self, 'fluence_optimizer') is None
                or 'optimized_dose' not in Datahub().optimization):

            # Log a message about the non-optimized plan
            self.logger.display_error(
                "Please optimize the treatment plan before evaluation!")

        else:

            # Initialize the DVH class
            self.dose_histogram = DVHEvaluator(
                dvh_type=self.evaluation.dvh_type,
                number_of_points=self.evaluation.number_of_points,
                display_segments=self.evaluation.display_segments)

            # Initialize the dosimetrics class
            self.dosimetrics = DosimetricsEvaluator(
                reference_volume=self.evaluation.reference_volume,
                reference_dose=self.evaluation.reference_dose,
                display_segments=self.evaluation.display_segments,
                display_metrics=self.evaluation.display_metrics)

            # Compute the dose-volume histogram from the optimized dose
            self.dose_histogram.evaluate(
                self.datahub.optimization['optimized_dose'])

            # Compute the dosimetrics from the optimized dose
            self.dosimetrics.evaluate(
                self.datahub.optimization['optimized_dose'])

            # Increment the state
            self.datahub.state = 4

    def visualize(
            self,
            parent=None):
        """
        Initialize and launch the visualization interface.

        Parameters
        ----------
        parent : None or object of class \
            :class:`~pyanno4rt.gui.windows._main_window.MainWindow`, \
                default=None
            The object used as a parent window for the visualization interface.
        """

        # Reset the treatment plan label in the datahub
        Datahub.label = self.configuration.label

        # Initialize the visualization interface
        self.visualizer = Visualizer(parent=parent)

        # Launch the visualization interface
        self.visualizer.launch()

    def compose(self):
        """Compose the treatment plan by cycling the workflow."""

        # Reset the treatment plan label in the datahub
        Datahub.label = self.configuration.label

        # Cycle the workflow
        self.configure()
        self.model()
        self.optimize()
        self.evaluate()
        self.visualize()

    def update(
            self,
            inputs):
        """
        Update the treatment plan by the input dictionary.

        Parameters
        ----------
        inputs : dict
            Dictionary with the update parameter(s).
        """

        # Loop over the items of the update dictionary
        for key, value in inputs.items():

            # Check if the key is in the configuration object
            if hasattr(self.configuration, key):

                # Check the configuration update
                self.configuration.check({key: value})

                # Update the configuration parameter value
                self.configuration.key = value

                # Check if the key is 'min_log_level'
                if key == 'min_log_level':

                    # Change the logging levels of all handlers
                    self.logger.change_log_levels(value)

            # Else, check if the key is in the optimization dictionary
            elif hasattr(self.optimization, key):

                # Check the optimization update
                self.optimization.check({key: value})

                # Update the optimization parameter value
                self.optimization.key = value

                # Check if the key is 'components'
                if key == 'components' and self.plan_generator is not None:

                    # Overwrite the components in the plan generator
                    self.plan_generator.components = value

                    # Update the components in the datahub
                    self.plan_generator.set_optimization_components(
                        verbose=False)

            # Else, check if the key is in the evaluation dictionary
            elif hasattr(self.evaluation, key):

                # Check the evaluation update
                self.evaluation.check({key: value})

                # Update the evaluation parameter value
                self.evaluation.key = value

            else:

                # Log a message about the missing key
                self.logger.display_warning(
                    f"The update parameter '{key}' is not part of the "
                    "treatment plan and will be ignored!")
