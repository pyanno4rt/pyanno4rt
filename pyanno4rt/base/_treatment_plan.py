"""Base treatment plan."""

# Author: Tim Ortkamp

# %% Internal package import

# Functional classes
from pyanno4rt.base import Configuration, Evaluation, Optimization
from pyanno4rt.logging import Logging, set_logger_name
from pyanno4rt.datahub import Datahub

# Treatment plan configuration
from pyanno4rt.patient import PatientHandler
from pyanno4rt.plan import PlanHandler
from pyanno4rt.dose import DoseHandler

# Treatment plan optimization
from pyanno4rt.optimization import FluenceOptimizer

# Treatment plan evaluation
from pyanno4rt.evaluation import DVH
from pyanno4rt.evaluation import Dosimetrics

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

    logging : object of class :class:`~pyanno4rt.logging._logging.Logging`
        The object used to print and store logging messages.

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

    dvh : None or object of class \
        :class:`~pyanno4rt.evaluation._dvh.DVH`
        The object used to evaluate the dose-volume histogram (DVH).

    dosimetrics : None or object of class \
        :class:`~pyanno4rt.evaluation._dosimetrics.Dosimetrics`
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
        self.logging = Logging(
            self.configuration.label, self.configuration.min_log_level)
        self.datahub = Datahub(self.configuration.label)
        self.patient_handler = None
        self.plan_handler = None
        self.dose_handler = None
        self.fluence_optimizer = None
        self.dvh = None
        self.dosimetrics = None
        self.visualizer = None

        # Initialize the state
        self.state = 0

    def configure(self):
        """Configure the treatment plan."""

        with self.context():

            # Validate the configuration parameters
            self.configuration.validate(vars(self.configuration))

            # Set the treatment plan label in the datahub
            Datahub.label = self.configuration.label

            # Initialize the patient handler
            self.patient_handler = PatientHandler()

            # Load the patient data
            self.patient_handler.load(self.configuration.imaging_path)

            # Initialize the plan handler
            self.plan_handler = PlanHandler(
                modality=self.configuration.modality,
                components=self.optimization.components)

            # Set the plan components
            self.plan_handler.set_components(self.patient_handler.segmentation)

            # Initialize the dose handler
            self.dose_handler = DoseHandler(
                dose_resolution=self.configuration.dose_resolution,
                number_of_fractions=self.configuration.number_of_fractions)

            # Generate the dose data
            self.dose_handler.generate(
                self.patient_handler.computed_tomography,
                self.plan_handler.modality,
                self.configuration.dose_matrix_path)

            # Resize the segments to the dose grid
            self.patient_handler.resize_segments(
                self.dose_handler.cube_dimensions)

            # Adjust the dose prescriptions to the fractionation
            self.plan_handler.adjust_parameters_for_fractionation(
                self.dose_handler.number_of_fractions)

            # Set the state
            self.state = 1

    def model(self):
        """Model the treatment plan outcome."""

        with self.context():

            # Set the treatment plan label in the datahub
            Datahub.label = self.configuration.label

            # Check if the plan has not been configured yet
            if None in (
                    self.patient_handler, self.plan_handler,
                    self.dose_handler):

                # Log a message about the non-configured plan
                self.logging.error(
                    "Please configure the treatment plan before modeling!")

            else:

                # Get the segmentation data
                segmentation = self.patient_handler.segmentation

                # Add the machine learning outcome models to the components
                apply(lambda component: component.add_model(), (
                    get_machine_learning_constraints(segmentation)
                    + get_machine_learning_objectives(segmentation)))

                # Set the state
                self.state = 2

    def optimize(self):
        """Optimize the treatment plan."""

        with self.context():

            # Set the treatment plan label in the datahub
            Datahub.label = self.configuration.label

            # Get the segmentation data
            segmentation = self.patient_handler.segmentation

            # Check if the plan has not been configured yet
            if None in (
                    self.patient_handler, self.plan_handler,
                    self.dose_handler):

                # Log a message about the non-configured plan
                self.logging.error(
                    "Please configure the treatment plan before optimization!")

            # Check if any machine learning component has not been modeled
            elif (None in (
                    component.model for component in
                    get_machine_learning_constraints(segmentation)
                    + get_machine_learning_objectives(segmentation))
                    or self.state < 2):

                # Log a message about the non-modeled component
                self.logging.error(
                    "Please add the machine learning models before "
                    "optimization!")

            else:

                # Validate the optimization parameters
                self.optimization.validate(vars(self.optimization))

                # Get the data handlers
                handlers = {
                    'patient_handler': self.patient_handler,
                    'plan_handler': self.plan_handler,
                    'dose_handler': self.dose_handler}

                # Initialize the fluence optimizer
                self.fluence_optimizer = FluenceOptimizer(
                    handlers=handlers,
                    method=self.optimization.method,
                    solver=self.optimization.solver,
                    algorithm=self.optimization.algorithm,
                    initial_strategy=self.optimization.initial_strategy,
                    initial_fluence=self.optimization.initial_fluence,
                    lower_variable_bounds=(
                        self.optimization.lower_variable_bounds),
                    upper_variable_bounds=(
                        self.optimization.upper_variable_bounds),
                    maximum_iterations=self.optimization.maximum_iterations,
                    tolerance=self.optimization.tolerance)

                # Solve the optimization problem
                self.fluence_optimizer.solve()

                # Set the state
                self.state = 3

    def evaluate(self):
        """Evaluate the treatment plan."""

        with self.context():

            # Set the treatment plan label in the datahub
            Datahub.label = self.configuration.label

            # Check if the plan has not been optimized yet
            if (self.fluence_optimizer is None
                    or self.fluence_optimizer.optimized_dose is None):

                # Log a message about the non-optimized plan
                self.logging.error(
                    "Please optimize the treatment plan before evaluation!")

            else:

                # Validate the evaluation parameters
                self.evaluation.validate(vars(self.evaluation))

                # Initialize the DVH
                self.dvh = DVH(
                    dvh_type=self.evaluation.dvh_type,
                    number_of_points=self.evaluation.number_of_points)

                # Compute the dose-volume histogram
                self.dvh.evaluate_segments(
                    self.patient_handler.segmentation,
                    self.fluence_optimizer.optimized_dose)

                # Initialize the dosimetrics
                self.dosimetrics = Dosimetrics(
                    self.evaluation.reference_volumes,
                    self.evaluation.reference_doses,
                    self.configuration.number_of_fractions)

                # Compute the dosimetrics
                self.dosimetrics.evaluate_segments(
                    self.patient_handler.segmentation,
                    self.fluence_optimizer.optimized_dose)

                # Set the state
                self.state = 4

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

        with self.context():

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

        # Initialize the update dictionaries
        configuration, optimization, evaluation = {}, {}, {}

        # Loop over the items of the input dictionary
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

                # Log a message about a missing key
                self.logging.warning(
                    f"The update parameter '{key}' is not part of the "
                    "treatment plan and will be ignored!")

        # Loop over the base objects and update dictionaries
        for base, update in (
                (self.configuration, configuration),
                (self.optimization, optimization),
                (self.evaluation, evaluation)):

            # Validate the update parameters
            base.validate(update)

            # Loop over the update items
            for key, value in update.items():

                # Update the attribute in the base object
                setattr(base, key, value)

                # Check if the key is 'min_log_level'
                if key == 'min_log_level':

                    # Change the logging levels of all handlers
                    self.logging.change_log_levels(value)

                # Check if the key is 'components'
                if key == 'components' and self.plan_handler is not None:

                    # Overwrite the components in the plan handler
                    self.plan_handler.components = value

                    # Update the components
                    self.plan_handler.set_components(
                        self.patient_handler.segmentation, verbose=False)

    def print_state(self):
        """Print the current state of the treatment plan."""

        # Define the states
        states = {
            0: 'initialized', 1: 'configured', 2: 'modeled', 3: 'optimized',
            4: 'evaluated'}

        return states[self.state]

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

    def context(self):
        """Set the context manager variables."""

        return set_logger_name(f'pyanno4rt - {self.configuration.label}')
