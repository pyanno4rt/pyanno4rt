"""Base treatment plan."""

# Author: Tim Ortkamp

# %% External package import

from copy import deepcopy
from warnings import filterwarnings

# %% Internal package import

# Functional classes
from pyanno4rt.base import Configuration, Evaluation, Optimization
from pyanno4rt.logging import Logging, set_logger_name

# Treatment plan configuration
from pyanno4rt.patient import PatientHandler
from pyanno4rt.plan import PlanHandler
from pyanno4rt.dose import DoseHandler

# Treatment plan outcome modeling
from pyanno4rt.learning import DataModelHandler

# Treatment plan optimization
from pyanno4rt.optimization import FluenceOptimizer

# Treatment plan evaluation
from pyanno4rt.evaluation import DVH
from pyanno4rt.evaluation import Dosimetrics

# Treatment plan visualization
from pyanno4rt.visualization import Visualizer

# Supporting functions
from pyanno4rt.tools import copycat, get_machine_learning_components, snapshot
from pyanno4rt.validation import validate_type

# %% Set package options

# Suppress pkg_resources deprecation warnings
filterwarnings("ignore", category=UserWarning, module="pkg_resources")
filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")

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

    data_model_handler : None or object of class \
        :class:`~pyanno4rt.learning._data_model_handler.DataModelHandler`
        The object used to handle the data-driven outcome models.

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
        self.patient_handler = None
        self.plan_handler = None
        self.dose_handler = None
        self.data_model_handler = None
        self.fluence_optimizer = None
        self.dvh = None
        self.dosimetrics = None
        self.visualizer = None

        # Initialize the state
        self.state = 0

    def configure(self):
        """Configure the treatment plan."""

        with self._context():

            # Validate the configuration parameters
            self.configuration.validate(vars(self.configuration))

            # Initialize the patient handler
            self.patient_handler = PatientHandler()

            # Initialize the plan handler
            self.plan_handler = PlanHandler(
                modality=self.configuration.modality,
                components=self.optimization.components)

            # Initialize the dose handler
            self.dose_handler = DoseHandler(
                dose_resolution=self.configuration.dose_resolution,
                number_of_fractions=self.configuration.number_of_fractions)

            # Load the patient data
            self.patient_handler.load(self.configuration.imaging_path)

            # Generate the dose data
            self.dose_handler.generate(
                self.patient_handler.computed_tomography,
                self.plan_handler.modality,
                self.configuration.dose_matrix_path)

            # Remove the segment overlaps
            self.patient_handler.remove_overlap(self.plan_handler.components)

            # Resize the segments to the dose grid
            self.patient_handler.resize_segments(
                self.dose_handler.cube_dimensions)

            # Add structure information to the components
            self.plan_handler.index_components(
                self.patient_handler.segmentation)

            # Adjust the dose prescriptions to the fractionation
            self.plan_handler.fractionate_components(
                self.dose_handler.number_of_fractions)

            # Set the state
            self.state = 1

    def model(self):
        """Model the treatment plan outcome."""

        with self._context():

            # Check if the plan has not been configured yet
            if None in (
                    self.patient_handler, self.plan_handler,
                    self.dose_handler):

                # Log a message about the non-configured plan
                self.logging.error(
                    "Please configure the treatment plan before modeling!")

            else:

                # Get the data handlers
                handlers = {
                    'patient_handler': self.patient_handler,
                    'plan_handler': self.plan_handler,
                    'dose_handler': self.dose_handler}

                # Initialize the data model handler
                self.data_model_handler = DataModelHandler(
                    handlers=handlers)

                # Load the datasets
                self.data_model_handler.load_datasets()

                # Set the feature calculators
                self.data_model_handler.set_calculators()

                # Fit the models
                self.data_model_handler.fit_models()

                # Inspect the models
                self.data_model_handler.inspect_models()

                # Evaluate the models
                self.data_model_handler.evaluate_models()

                # Set the state
                self.state = 2

    def optimize(self):
        """Optimize the treatment plan."""

        with self._context():

            # Check if the plan has not been configured yet
            if None in (
                    self.patient_handler, self.plan_handler,
                    self.dose_handler):

                # Log a message about the non-configured plan
                self.logging.error(
                    "Please configure the treatment plan before optimization!")

            # Check if the outcome has not been modeled yet
            elif (self.data_model_handler is None and
                  len(get_machine_learning_components(
                      self.plan_handler.components)) > 0):

                # Log a message about the non-modeled component
                self.logging.error(
                    "Please integrate the machine learning outcome models "
                    "before optimization!")

            else:

                # Validate the optimization parameters
                self.optimization.validate(vars(self.optimization))

                # Get the data handlers
                handlers = {
                    'patient_handler': self.patient_handler,
                    'plan_handler': self.plan_handler,
                    'dose_handler': self.dose_handler,
                    'data_model_handler': self.data_model_handler}

                # Initialize the fluence optimizer
                self.fluence_optimizer = FluenceOptimizer(handlers=handlers)

                # Initialize the optimization problem
                self.fluence_optimizer.initialize_problem(
                    self.optimization.method,
                    self.optimization.initial_strategy,
                    self.optimization.initial_fluence,
                    self.optimization.lower_variable_bounds,
                    self.optimization.upper_variable_bounds)

                # Initialize the solver
                self.fluence_optimizer.initialize_solver(
                    self.optimization.solver,
                    self.optimization.algorithm,
                    self.optimization.maximum_iterations,
                    self.optimization.tolerance)

                # Solve the optimization problem
                self.fluence_optimizer.solve()

                # Set the state
                self.state = 3

    def evaluate(self):
        """Evaluate the treatment plan."""

        with self._context():

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

                # Initialize the dosimetrics
                self.dosimetrics = Dosimetrics(
                    self.evaluation.reference_volumes,
                    self.evaluation.reference_doses,
                    self.configuration.number_of_fractions)

                # Compute the dose-volume histogram
                self.dvh.evaluate_segments(
                    self.patient_handler.segmentation,
                    self.fluence_optimizer.optimized_dose)

                # Compute the dosimetrics
                self.dosimetrics.evaluate_segments(
                    self.patient_handler.segmentation,
                    self.plan_handler.components,
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

        with self._context():

            # Initialize the visualizer
            self.visualizer = Visualizer(treatment_plan=self, parent=parent)

            # Set the position of the window
            self.visualizer.position()

            # Show the window
            self.visualizer.launch()

    def compose(self):
        """Compose the treatment plan."""

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
            include_optimum=False,
            include_tracks=False,
            anonymize=False):
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

        include_tracks : bool, default=False
            Indicator for the storage of the component tracker.

        anonymize : bool, default=False
            Indicator for the anonymization of the file paths.

        Notes
        -----
        See :func:`~pyanno4rt.tools._snapshot.snapshot` for details.
        """

        # Take a snapshot
        snapshot(
            deepcopy(self), path, include_patient_data, include_dose_matrix,
            include_model_data, include_optimum, include_tracks, anonymize)

    @staticmethod
    def load(
            path,
            ignore_optimum=False):
        """
        Load a treatment plan from a snapshot.

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

    def __deepcopy__(
            self,
            memo):
        """
        Return a deep copy of the object.

        Parameters
        ----------
        memo : dict
            Dictionary of objects already copied.

        Returns
        -------
        object of class :class:`~pyanno4rt.base._treatment_plan.TreatmentPlan`
            The object used to represent the (pickable) treatment plan.
        """

        # Get the instance class
        cls = self.__class__

        # Create a new instance
        result = cls.__new__(cls)

        # Store the instance
        memo[id(self)] = result

        # Loop over the instance attributes
        for key, value in self.__dict__.items():

            # Check if the visualizer attribute is considered
            if key == 'visualizer':

                # Skip the non-picklable attribute
                setattr(result, key, None)

            else:

                # Pickle the attribute
                setattr(result, key, deepcopy(value, memo))

        return result

    def _context(self):
        """Set the context manager variables."""

        return set_logger_name(f'pyanno4rt - {self.configuration.label}')
