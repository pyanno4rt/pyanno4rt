"""Base treatment plan."""

# Author: Tim Ortkamp

# %% Internal package import

# Functional classes
from pyanno4rt.base import Configuration, Evaluation, Optimization
from pyanno4rt.logging import Logger
from pyanno4rt.datahub import Datahub

# Treatment plan configuration
from pyanno4rt.patient import PatientHandler
from pyanno4rt.plan import PlanHandler
from pyanno4rt.dose import DoseHandler

# Treatment plan optimization
from pyanno4rt.optimization import FluenceOptimizer

# Treatment plan evaluation
from pyanno4rt.evaluation import DVHEvaluator
from pyanno4rt.evaluation import DosimetricsEvaluator

# Treatment plan visualization
from pyanno4rt.visualization import Visualizer

# Supporting functions
from pyanno4rt.tools import (
    apply, copycat, get_machine_learning_constraints,
    get_machine_learning_objectives, snapshot)
from pyanno4rt.validation import validate_type

# %% Class definition


class TreatmentPlan():
    """
    Base treatment plan class.

    This class enables configuration, optimization, evaluation, and \
    visualization of individual IMRT treatment plans.

    Parameters
    ----------
    configuration : object of class \
        :class:`~pyanno4rt.base._configuration.Configuration` or dict
        The object used to handle the plan configuration parameters.

    optimization : object of class \
        :class:`~pyanno4rt.base._optimization.Optimization` or dict
        The object used to handle the plan optimization parameters.

    evaluation : object of class \
        :class:`~pyanno4rt.base._evaluation.Evaluation` or dict
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

    logger : object of class :class:`~pyanno4rt.logging._logger.Logger`
        The object used to print and store logging messages.

    datahub : object of class :class:`~pyanno4rt.datahub._datahub.Datahub`
        The object used to manage and distribute information units.

    patient_handler : None or object of class \
        :class:`~pyanno4rt.patient._patient_handler.PatientHandler`
        The object used to handle the patient imaging data.

    plan_handler : None or object of class \
        :class:`~pyanno4rt.plan._plan_handler.PlanHandler`
        The object used to handle the plan parameters.

    dose_handler : None or object of class \
        :class:`~pyanno4rt.dose._dose_handler.DoseHandler`
        The object used to handle the dose parameters.

    fluence_optimizer : None or object of class \
        :class:`~pyanno4rt.optimization._fluence_optimizer.FluenceOptimizer`
        The object used to solve the fluence optimization problem.

    dose_histogram : None or object of class \
        :class:`~pyanno4rt.evaluation._dvh_evaluator.DVHEvaluator`
        The object used to evaluate the dose-volume histogram (DVH).

    dosimetrics : None or object of class \
        :class:`~pyanno4rt.evaluation._dosimetrics_evaluator.DosimetricsEvaluator`
        The object used to evaluate the dosimetrics.

    visualizer : None or object of class \
        :class:`~pyanno4rt.visualization._visualizer.Visualizer`
        The object used to visualize the treatment plan.

    Example
    -------
    Our Read the Docs page (https://pyanno4rt.readthedocs.io/en/latest/) \
    features step-by-step examples for the application of the package. You \
    will also find code snippets there, e.g. for the optimization components.
    """

    def __init__(
            self,
            configuration,
            optimization,
            evaluation):

        # Validate the input arguments
        validate_type('configuration', configuration, (dict, Configuration))
        validate_type('optimization', optimization, (dict, Optimization))
        validate_type('evaluation', evaluation, (dict, Evaluation))

        # Initialize the plan configuration attribute
        self.configuration = (
            configuration if not isinstance(configuration, dict)
            else Configuration.from_dict(configuration))

        # Initialize the plan optimization attribute
        self.optimization = (
            optimization if not isinstance(optimization, dict)
            else Optimization.from_dict(optimization))

        # Initialize the plan evaluation attribute
        self.evaluation = (
            evaluation if not isinstance(evaluation, dict)
            else Evaluation.from_dict(evaluation))

        # Initialize the plan subclasses
        self.logger = Logger(
            self.configuration.label, self.configuration.min_log_level)
        self.datahub = Datahub(self.configuration.label, self.logger)
        self.patient_handler = None
        self.plan_handler = None
        self.dose_handler = None
        self.fluence_optimizer = None
        self.dose_histogram = None
        self.dosimetrics = None
        self.visualizer = None

        # Initialize the state
        self.datahub.state = 0

    def configure(self):
        """Configure the treatment plan."""

        # Validate the configuration parameters
        self.configuration.validate(vars(self.configuration))

        # Set the treatment plan label in the datahub
        Datahub.label = self.configuration.label

        # Initialize the patient handler
        self.patient_handler = PatientHandler()

        # Load the patient data
        self.patient_handler.load(path=self.configuration.imaging_path)

        # Initialize the plan handler
        self.plan_handler = PlanHandler(
            modality=self.configuration.modality,
            components=self.optimization.components)

        # Generate the plan data
        self.plan_handler.generate()

        # Initialize the dose handler
        self.dose_handler = DoseHandler(
            dose_resolution=self.configuration.dose_resolution,
            number_of_fractions=self.configuration.number_of_fractions)

        # Load the dose data
        self.dose_handler.load(
            dose_matrix_path=self.configuration.dose_matrix_path)

        # Set the state
        self.datahub.state = 1

    def model(self):
        """Model the treatment plan outcome."""

        # Set the treatment plan label in the datahub
        Datahub.label = self.configuration.label

        # Check if the plan has not been configured yet
        if None in (
                self.patient_handler, self.plan_handler, self.dose_handler):

            # Log a message about the non-configured plan
            self.logger.display_error(
                "Please configure the treatment plan before modeling!")

        else:

            # Get the segmentation data
            segmentation = Datahub().segmentation

            # Add the machine learning outcome models to the components
            apply(lambda component: component.add_model(), (
                get_machine_learning_constraints(segmentation)
                + get_machine_learning_objectives(segmentation)))

            # Set the state
            self.datahub.state = 2

    def optimize(self):
        """Optimize the treatment plan."""

        # Set the treatment plan label in the datahub
        Datahub.label = self.configuration.label

        # Get the segmentation data
        segmentation = Datahub().segmentation

        # Check if the plan has not been configured yet
        if None in (
                self.patient_handler, self.plan_handler, self.dose_handler):

            # Log a message about the non-configured plan
            self.logger.display_error(
                "Please configure the treatment plan before optimization!")

        # Check if any machine learning component has not been modeled
        elif None in (
                component.model for component in
                get_machine_learning_constraints(segmentation)
                + get_machine_learning_objectives(segmentation)):

            # Log a message about the non-modeled component
            self.logger.display_error(
                "Please add the machine learning models before optimization!")

        else:

            # Validate the optimization parameters
            self.optimization.validate(vars(self.optimization))

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

            # Set the state
            self.datahub.state = 3

    def evaluate(self):
        """Evaluate the treatment plan."""

        # Set the treatment plan label in the datahub
        Datahub.label = self.configuration.label

        # Check if the plan has not been optimized yet
        if (self.fluence_optimizer is None
                or 'optimized_dose' not in Datahub().optimization):

            # Log a message about the non-optimized plan
            self.logger.display_error(
                "Please optimize the treatment plan before evaluation!")

        else:

            # Validate the evaluation parameters
            self.evaluation.validate(vars(self.evaluation))

            # Initialize the DVH evaluator
            self.dose_histogram = DVHEvaluator(
                dvh_type=self.evaluation.dvh_type,
                number_of_points=self.evaluation.number_of_points)

            # Compute the dose-volume histogram
            self.dose_histogram.evaluate(
                self.datahub.optimization['optimized_dose'])

            # Initialize the dosimetrics evaluator
            self.dosimetrics = DosimetricsEvaluator(
                reference_volume=self.evaluation.reference_volume,
                reference_dose=self.evaluation.reference_dose)

            # Compute the dosimetrics
            self.dosimetrics.evaluate(
                self.datahub.optimization['optimized_dose'])

            # Set the state
            self.datahub.state = 4

    def visualize(
            self,
            parent=None):
        """
        Visualize the treatment plan.

        Parameters
        ----------
        parent : None or object of class \
            :class:`~pyanno4rt.gui.windows._main_window.MainWindow`, \
                default=None
            The object used as a parent window for the visualizer.
        """

        # Set the treatment plan label in the datahub
        Datahub.label = self.configuration.label

        # Initialize the visualizer
        self.visualizer = Visualizer(treatment_plan=self, parent=parent)

        # Set the position of the window
        self.visualizer.position()

        # Show the window
        self.visualizer.launch()

    def compose(self):
        """Compose the treatment plan."""

        # Set the treatment plan label in the datahub
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
        Update the treatment plan.

        Parameters
        ----------
        inputs : dict
            Dictionary with the update parameter(s).
        """

        # Initialize the base dictionaries
        configuration, optimization, evaluation = {}, {}, {}

        # Loop over the items of the update dictionary
        for key, value in inputs.items():

            # Check if the key is in the configuration object
            if hasattr(self.configuration, key):

                # Add the item to the configuration base
                configuration |= {key: value}

            # Else, check if the key is in the optimization object
            elif hasattr(self.optimization, key):

                # Add the item to the optimization base
                optimization |= {key: value}

            # Else, check if the key is in the evaluation object
            elif hasattr(self.evaluation, key):

                # Add the item to the evaluation base
                evaluation |= {key: value}

            else:

                # Log a message about the missing key
                self.logger.display_warning(
                    f"The update parameter '{key}' is not part of the "
                    "treatment plan and will be ignored!")

        # Loop over the base objects and dictionaries
        for base_object, update in (
                (self.configuration, configuration),
                (self.optimization, optimization),
                (self.evaluation, evaluation)):

            # Validate the update parameters
            base_object.validate(update)

            # Loop over the update items
            for key, value in update.items():

                # Update the attribute in the base object
                setattr(base_object, key, value)

                # Check if the key is 'min_log_level'
                if key == 'min_log_level':

                    # Change the logging levels of all handlers
                    self.logger.change_log_levels(value)

                # Check if the key is 'components'
                if key == 'components' and self.plan_handler is not None:

                    # Overwrite the components in the plan handler
                    self.plan_handler.components = value

                    # Update the components
                    self.plan_handler.set_optimization_components(
                        verbose=False)

    def state(self):
        """Get the current state of the treatment plan."""

        # Set the treatment plan label in the datahub
        Datahub.label = self.configuration.label

        # Define the states
        states = {
            0: 'initialized', 1: 'configured', 2: 'modeled', 3: 'optimized',
            4: 'evaluated'}

        return states[self.datahub.state]

    def save(
            self,
            path,
            include_patient_data=False,
            include_dose_matrix=False,
            include_model_data=False,
            include_optimum=False):
        """
        Save a treatment plan.

        Parameters
        ----------
        path : str
            Directory path for the snapshot (folder).

        include_patient_data : bool, default=False
            Indicator for the storage of the external patient data, i.e., \
            CT and segmentation data.

        include_dose_matrix : bool, default=False
            Indicator for the storage of the dose-influence matrix.

        include_model_data : bool, default=False
            Indicator for the storage of the outcome model-related datasets.

        include_optimum : bool, default=False
            Indicator for the storage of the optimized fluence array.

        Notes
        -----
        See :func:`~pyanno4rt.tools._snapshot.snapshot` for details.
        """

        # Take a snapshot
        snapshot(
            self, path, include_patient_data, include_dose_matrix,
            include_model_data, include_optimum)

    @staticmethod
    def load(
            path,
            ignore_optimum=False):
        """
        Load a treatment plan from a snapshot. See \
        :func:`~pyanno4rt.tools._copycat.copycat` for details.

        Parameters
        ----------
        path : str
            Directory path of the snapshot.

        ignore_optimum : bool
            Indicator for ignoring the optimal fluence file (if available).

        Returns
        -------
        object of class :class:`~pyanno4rt.base._treatment_plan.TreatmentPlan`
            The object used to represent the treatment plan.

        Notes
        -----
        See :func:`~pyanno4rt.tools._copycat.copycat` for details.
        """

        return copycat(TreatmentPlan, path, ignore_optimum)
