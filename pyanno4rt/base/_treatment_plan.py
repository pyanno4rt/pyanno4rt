"""Base treatment plan."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% Internal package import

# Functional classes
from pyanno4rt.logging import Logger
from pyanno4rt.datahub import Datahub
from pyanno4rt.input_check import InputChecker

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
from pyanno4rt.tools import apply

# %% Class definition


class TreatmentPlan():
    """
    Base treatment plan class.

    This class enables configuration, optimization, evaluation, and \
    visualization of individual IMRT treatment plans. It therefore provides a \
    simple, but extensive interface using input dictionaries for the \
    different parameter groups.

    Parameters
    ----------
    configuration : dict
        Dictionary with the treatment plan configuration parameters.

        - label : str
            Unique identifier for the treatment plan.

            .. note:: Uniqueness of the label is important because it \
                prevents overwriting processes between different treatment \
                plan instances by isolating their datahubs, logging channels \
                and general storage paths.

        - min_log_level : {'debug', 'info', 'warning', 'error, 'critical'}, \
                           default='info'
            Minimum logging level.

        - modality : {'photon', 'proton'}
            Treatment modality, needs to be consistent with the dose \
            calculation inputs.

            .. note:: If the modality is 'photon', \
                :class:`~pyanno4rt.optimization.projections._dose_projection.DoseProjection`\
                with neutral RBE of 1.0 is automatically applied, whereas for \
                the modality 'proton', \
                :class:`~pyanno4rt.optimization.projections._constant_rbe_projection.ConstantRBEProjection`\
                with constant RBE of 1.1 is used.

        - number_of_fractions : int
            Number of fractions according to the treatment scheme.

        - imaging_path : str
            Path to the CT and segmentation data.

            .. note:: It is assumed that CT and segmentation data are \
                included in a single file (.mat or .p) or a series of files \
                (.dcm), whose content follows the pyanno4rt data structure.

        - target_imaging_resolution : None or list, default=None
            Imaging resolution for post-processing interpolation of the CT \
            and segmentation data, only used if a list is passed.

        - dose_matrix_path : str
            Path to the dose-influence matrix file (.mat or .npy).

        - dose_resolution : list
            Size of the dose grid in [`mm`] per dimension, needs to be \
            consistent with the dose calculation inputs.

    optimization : dict
        Dictionary with the treatment plan optimization parameters.

        - components : dict
            Optimization components for each segment of interest, i.e., \
            objective functions and constraints.

            .. note:: The declaration scheme for a single component is

                {<segment>: {'type': <1>, 'instance': {'class': <2>, \
                                                       'parameters': <3>}

                - <1>: 'objective' or 'constraint'
                - <2>: component label (see note below)
                - <3> parameter dictionary for the component (see the \
                  component classes for details)

                Multiple objectives and/or constraints can be assigned by \
                passing a list of dictionaries for each segment of interest.

                The following components are currently available:

                - 'Decision Tree NTCP' \
                    :class:`~pyanno4rt.optimization.components._decision_tree_ntcp.DecisionTreeNTCP`
                - 'Decision Tree TCP' \
                    :class:`~pyanno4rt.optimization.components._decision_tree_tcp.DecisionTreeTCP`
                - 'Dose Uniformity' \
                    :class:`~pyanno4rt.optimization.components._dose_uniformity.DoseUniformity`
                - 'Equivalent Uniform Dose' \
                    :class:`~pyanno4rt.optimization.components._equivalent_uniform_dose.EquivalentUniformDose`
                - 'K-Nearest Neighbors NTCP' \
                    :class:`~pyanno4rt.optimization.components._k_nearest_neighbors_ntcp.KNeighborsNTCP`
                - 'K-Nearest Neighbors TCP' \
                    :class:`~pyanno4rt.optimization.components._k_nearest_neighbors_tcp.KNeighborsTCP`
                - 'Logistic Regression NTCP' \
                    :class:`~pyanno4rt.optimization.components._logistic_regression_ntcp.LogisticRegressionNTCP`
                - 'Logistic Regression TCP' \
                    :class:`~pyanno4rt.optimization.components._logistic_regression_tcp.LogisticRegressionTCP`
                - 'LQ Poisson TCP' \
                    :class:`~pyanno4rt.optimization.components._lq_poisson_tcp.LQPoissonTCP`
                - 'Lyman-Kutcher-Burman NTCP' \
                    :class:`~pyanno4rt.optimization.components._lyman_kutcher_burman_ntcp.LymanKutcherBurmanNTCP`
                - 'Maximum DVH' \
                    :class:`~pyanno4rt.optimization.components._maximum_dvh.MaximumDVH`
                - 'Mean Dose' \
                    :class:`~pyanno4rt.optimization.components._mean_dose.MeanDose`
                - 'Minimum DVH' \
                    :class:`~pyanno4rt.optimization.components._minimum_dvh.MinimumDVH`
                - 'Naive Bayes NTCP' \
                    :class:`~pyanno4rt.optimization.components._naive_bayes_ntcp.NaiveBayesNTCP`
                - 'Naive Bayes TCP' \
                    :class:`~pyanno4rt.optimization.components._naive_bayes_tcp.NaiveBayesTCP`
                - 'Neural Network NTCP' \
                    :class:`~pyanno4rt.optimization.components._neural_network_ntcp.NeuralNetworkNTCP`
                - 'Neural Network TCP' \
                    :class:`~pyanno4rt.optimization.components._neural_network_tcp.NeuralNetworkTCP`
                - 'Random Forest NTCP' \
                    :class:`~pyanno4rt.optimization.components._random_forest_ntcp.RandomForestNTCP`
                - 'Random Forest TCP' \
                    :class:`~pyanno4rt.optimization.components._random_forest_tcp.RandomForestTCP`
                - 'Squared Deviation' \
                    :class:`~pyanno4rt.optimization.components._squared_deviation.SquaredDeviation`
                - 'Squared Overdosing' \
                    :class:`~pyanno4rt.optimization.components._squared_overdosing.SquaredOverdosing`
                - 'Squared Underdosing' \
                    :class:`~pyanno4rt.optimization.components._squared_underdosing.SquaredUnderdosing`
                - 'Support Vector Machine NTCP' \
                    :class:`~pyanno4rt.optimization.components._support_vector_machine_ntcp.SupportVectorMachineNTCP`
                - 'Support Vector Machine TCP' \
                    :class:`~pyanno4rt.optimization.components._support_vector_machine_tcp.SupportVectorMachineTCP`

        - method : {'lexicographic', 'pareto', 'weighted-sum'}, \
            default='weighted-sum'
            Single- or multi-criteria optimization method, see the classes \
            :class:`~pyanno4rt.optimization.methods._lexicographic_optimization.LexicographicOptimization`\
            :class:`~pyanno4rt.optimization.methods._pareto_optimization.ParetoOptimization`\
            :class:`~pyanno4rt.optimization.methods._weighted_sum_optimization.WeightedSumOptimization`.

            - 'lexicographic' : sequential optimization based on a \
                preference order
            - 'pareto' : parallel optimization based on the criterion of \
                pareto optimality
            - 'weighted-sum' : parallel optimization based on a weighted-sum \
                scalarization of the objective function

        - solver : {'proxmin', 'pymoo', 'pypop7', 'scipy'}, default='scipy'
            Python package to be used for solving the optimization problem, \
            see the classes \
            :class:`~pyanno4rt.optimization.solvers._proxmin_solver.ProxminSolver`\
            :class:`~pyanno4rt.optimization.solvers._pymoo_solver.PymooSolver`\
            :class:`~pyanno4rt.optimization.solvers._pypop7_solver.PyPop7Solver`\
            :class:`~pyanno4rt.optimization.solvers._scipy_solver.SciPySolver`.

            - 'proxmin' : proximal algorithms provided by Proxmin
            - 'pymoo' : multi-objective algorithms provided by Pymoo
            - 'pypop7': population-based algorithms provided by PyPop7
            - 'scipy' : local algorithms provided by SciPy

            .. note:: The 'lexicographic' method currently only works with \
                'scipy', while the 'pareto' method only works with 'pymoo'.

        - algorithm : str
            Solution algorithm from the chosen solver:

            - solver='proxmin' : {'admm', 'pgm', 'sdmm'}, default='pgm'

                - 'admm' : alternating direction method of multipliers
                - 'pgm' : proximal gradient method
                - 'sdmm' : simultaneous direction method of multipliers

            - solver='pymoo' : {'NSGA3'}, default='NSGA3'

                - 'NSGA3' : non-dominated sorting genetic algorithm III

            - solver='pypop7' : {'LMCMA', 'LMMAES'}, default='LMCMA'

                - 'LMCMA' : limited-memory covariance matrix adaptation
                - 'LMMAES' : limited-memory matrix adaptation evolution \
                    strategy

            - solver='scipy' : {'L-BFGS-B', 'TNC', 'trust-constr'}, \
                default='L-BFGS-B'

                - 'L-BFGS-B' : bounded limited memory \
                    Broyden-Fletcher-Goldfarb-Shanno method
                - 'TNC' : truncated Newton method
                - 'trust-constr' : trust-region constrained method

            .. note:: Constraints are currently only supported by 'NSGA3' \
                and 'trust-constr'.

        - initial_strategy : {'data-medoid', 'target-coverage', \
                              'warm-start'}, default='target-coverage'
            Initialization strategy for the fluence vector (see the class \
            :class:`~pyanno4rt.optimization.initializers._fluence_initializer.FluenceInitializer`).

            - 'data-medoid' : fluence vector initialization with respect to \
                data medoid points
            - 'target-coverage' : fluence vector initialization with respect \
                to tumor coverage
            - 'warm-start' : fluence vector initialization with respect to a \
                reference optimal point

            .. note:: Data-medoid initialization works best for a single \
                dataset or multiple datasets with a high degree of \
                similarity. Otherwise, the initial fluence vector may lose \
                its individual representativeness.

        - initial_fluence_vector : None or list, default=None
            User-defined initial fluence vector for the optimization problem, \
            only used if initial_strategy='warm-start' (see the class \
            :class:`~pyanno4rt.optimization.initializers._fluence_initializer.FluenceInitializer`).

        - lower_variable_bounds : None, int, float, or list, default=0
            Lower bound(s) on the decision variables.

        - upper_variable_bounds : None, int, float, or list, default=None
            Upper bound(s) on the decision variables.

        .. note:: There are two options to set lower and upper bounds for the \
            variables:

                1) Passing a single numeric value translates into uniform \
                    bounds across all variables (where None for the lower \
                    and/or upper bound indicates infinity bounds)
                2) Passing a list translates into non-uniform bounds (here, \
                    the length of the list needs to be equal to the number of \
                    decision variables)

        - max_iter : int, default=500
            Maximum number of iterations taken for the solver to converge.

        - tolerance : float, default=1e-3
            Precision goal for the objective function value.

    evaluation : None or dict, default=None
        Dictionary with the treatment plan evaluation parameters.

        - dvh_type : {'cumulative', 'differential'}, default=cumulative'
            Type of DVH to be evaluated.

        - number_of_points : int, default=1000
            Number of (evenly-spaced) points for which to evaluate the DVH.

        - reference_volume : list, default=[2, 5, 50, 95, 98]
            Reference volumes for which to evaluate the inverse DVH values.

        - reference_dose : list, default=[]
            Reference dose values for which to evaluate the DVH values.

            .. note:: If the default value is used, reference dose \
                levels will be determined automatically.

        - display_segments : list, default=[]
            Names of the segmented structures to be displayed.

            .. note:: If the default value is used, all segments will \
                be displayed.

        - display_metrics : list, default=[]
            Names of the plan evaluation metrics to be displayed.

            .. note:: If the default value is used, all metrics will be \
                displayed.

                The following metrics are currently available:

                - 'mean': mean dose
                - 'std': standard deviation of the dose
                - 'max': maximum dose
                - 'min': minimum dose
                - 'Dx': dose quantile(s) for level x (reference_volume)
                - 'Vx': volume quantile(s) for level x (reference_dose)
                - 'CI': conformity index
                - 'HI': homogeneity index

    Attributes
    ----------
    configuration : dict
        See 'Parameters'.

    optimization : dict
        See 'Parameters'.

    evaluation : dict
        See 'Parameters'.

    input_checker : object of class \
        :class:`~pyanno4rt.input_check._input_checker.InputChecker`
        The object used to approve the input dictionaries.

    logger : None or object of class \
        :class:`~pyanno4rt.logging._logger.Logger`
        The internal object used to print and store logging messages.

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
            evaluation=None):

        # Check if the evaluation dictionary has not been specified
        if evaluation is None:

            # Initialize the evaluation dictionary to its default
            evaluation = {}

        # Initialize the input checker
        self.input_checker = InputChecker()

        # Approve the input types
        self.input_checker.approve({
            'configuration': configuration,
            'optimization': optimization,
            'evaluation': evaluation})

        # Initialize the configuration parameter dictionary
        self.configuration = {
            'label': configuration.get('label'),
            'min_log_level': configuration.get('min_log_level', 'info'),
            'modality': configuration.get('modality'),
            'number_of_fractions': configuration.get(
                'number_of_fractions', 30),
            'imaging_path': configuration.get('imaging_path'),
            'target_imaging_resolution': configuration.get(
                'target_imaging_resolution'),
            'dose_matrix_path': configuration.get('dose_matrix_path'),
            'dose_resolution': configuration.get('dose_resolution')
            }

        # Initialize the optimization parameter dictionary
        self.optimization = {
            'components': optimization.get('components'),
            'method': optimization.get('method', 'weighted-sum'),
            'solver': optimization.get('solver', 'scipy'),
            'algorithm': optimization.get(
                'algorithm',
                'pgm' if optimization.get('solver') == 'proxmin'
                else 'NSGA3' if optimization.get('solver') == 'pymoo'
                else 'LMCMA' if optimization.get('solver') == 'pypop7'
                else 'L-BFGS-B'),
            'initial_strategy': optimization.get(
                'initial_strategy', 'target-coverage'),
            'initial_fluence_vector': optimization.get(
                'initial_fluence_vector', None),
            'lower_variable_bounds': optimization.get(
                'lower_variable_bounds', 0),
            'upper_variable_bounds': optimization.get(
                'upper_variable_bounds', None),
            'max_iter': optimization.get('max_iter', 500),
            'tolerance': optimization.get('tolerance', 1e-3)
            }

        # Initialize the treatment plan evaluation dictionary
        self.evaluation = {
            'dvh_type': evaluation.get('dvh_type', 'cumulative'),
            'number_of_points': evaluation.get('number_of_points', 1000),
            'reference_volume': evaluation.get(
                'reference_volume', [2, 5, 50, 95, 98]),
            'reference_dose': evaluation.get('reference_dose', []),
            'display_segments': evaluation.get('display_segments', []),
            'display_metrics': evaluation.get('display_metrics', [])
            }

        # Approve the input values
        apply(self.input_checker.approve,
              (self.configuration, self.optimization, self.evaluation))

        # Initialize the instance attributes
        self.logger = None
        self.datahub = None
        self.patient_loader = None
        self.plan_generator = None
        self.dose_info_generator = None
        self.fluence_optimizer = None
        self.dose_histogram = None
        self.dosimetrics = None
        self.visualizer = None

    def configure(self):
        """Initialize the configuration classes and process the input data."""

        # Initialize the logger
        self.logger = Logger(
            self.configuration['label'], self.configuration['min_log_level'])

        # Initialize the datahub
        self.datahub = Datahub(
            self.configuration['label'], self.input_checker, self.logger)

        # Initialize the patient loader
        self.patient_loader = PatientLoader(
            imaging_path=self.configuration['imaging_path'],
            target_imaging_resolution=self.configuration[
                'target_imaging_resolution'])

        # Initialize the plan generator
        self.plan_generator = PlanGenerator(
            modality=self.configuration['modality'])

        # Initialize the dose information generator
        self.dose_info_generator = DoseInfoGenerator(
            number_of_fractions=self.configuration['number_of_fractions'],
            dose_matrix_path=self.configuration['dose_matrix_path'],
            dose_resolution=self.configuration['dose_resolution'])

        # Load the patient data
        self.patient_loader.load()

        # Generate the plan information
        self.plan_generator.generate()

        # Generate the dose information
        self.dose_info_generator.generate()

    def optimize(self):
        """
        Initialize the fluence optimizer and solve the problem.

        Raises
        ------
        AttributeError
            If the treatment plan has not been configured yet.
        """

        # Check if any required attribute is missing
        if any(getattr(self, attribute) is None for attribute in (
                'logger', 'datahub', 'input_checker', 'patient_loader',
                'plan_generator', 'dose_info_generator')):

            # Check if the logger has been initialized
            if self.logger:

                # Log a message about the attribute error
                self.logger.display_error(
                    "Please configure the treatment plan before optimization!")

            # Raise an error to indicate a missing attribute
            raise AttributeError(
                "Please configure the treatment plan before optimization!")

        # Reset the treatment plan label in the datahub
        Datahub.label = self.configuration['label']

        # Initialize the fluence optimizer
        self.fluence_optimizer = FluenceOptimizer(
            components=self.optimization['components'],
            method=self.optimization['method'],
            solver=self.optimization['solver'],
            algorithm=self.optimization['algorithm'],
            initial_strategy=self.optimization['initial_strategy'],
            initial_fluence_vector=self.optimization['initial_fluence_vector'],
            lower_variable_bounds=self.optimization['lower_variable_bounds'],
            upper_variable_bounds=self.optimization['upper_variable_bounds'],
            max_iter=self.optimization['max_iter'],
            tolerance=self.optimization['tolerance'])

        # Solve the optimization problem with the optimizer
        self.fluence_optimizer.solve()

    def evaluate(self):
        """
        Initialize the evaluation classes and compute the plan metrics.

        Raises
        ------
        AttributeError
            If the treatment plan has not been optimized yet.
        """

        # Check if the 'fluence_optimizer' attribute is missing
        if getattr(self, 'fluence_optimizer') is None:

            # Check if the logger has been initialized
            if self.logger:

                # Log a message about the attribute error
                self.logger.display_error(
                    "Please optimize the treatment plan before evaluation!")

            # Raise an error to indicate the missing attribute
            raise AttributeError(
                "Please optimize the treatment plan before evaluation!")

        # Reset the treatment plan label in the datahub
        Datahub.label = self.configuration['label']

        # Initialize the DVH class
        self.dose_histogram = DVHEvaluator(
            dvh_type=self.evaluation['dvh_type'],
            number_of_points=self.evaluation['number_of_points'],
            display_segments=self.evaluation['display_segments'])

        # Initialize the dosimetrics class
        self.dosimetrics = DosimetricsEvaluator(
            reference_volume=self.evaluation['reference_volume'],
            reference_dose=self.evaluation['reference_dose'],
            display_segments=self.evaluation['display_segments'],
            display_metrics=self.evaluation['display_metrics'])

        # Check if the dose has been optimized
        if self.datahub.optimization['optimized_dose'] is not None:

            # Compute the dose-volume histogram from the optimized dose
            self.dose_histogram.evaluate(
                self.datahub.optimization['optimized_dose'])

            # Compute the dosimetrics from the optimized dose
            self.dosimetrics.evaluate(
                self.datahub.optimization['optimized_dose'])

    def visualize(
            self,
            parent=None):
        """
        Initialize the visualization interface and launch it.

        Parameters
        ----------
        parent : None or object of class \
            :class:`~pyanno4rt.gui.windows._main_window.MainWindow`, \
                default=None
            The object used as a parent window for the visualization interface.

        Raises
        ------
        AttributeError
            If the treatment plan has not been optimized (and evaluated) yet.
        """

        # Check if any required attribute is missing
        if all(getattr(self, attribute) is None for attribute in (
                'fluence_optimizer', 'dose_histogram', 'dosimetrics')):

            # Check if the logger has been initialized
            if self.logger:

                # Log a message about the attribute error
                self.logger.display_error(
                    "Please optimize (and optionally evaluate) the treatment "
                    "plan before launching the visualization interface!")

            # Raise an error to indicate a missing attribute
            raise AttributeError(
                "Please optimize (and optionally evaluate) the treatment plan \
                before launching the visualization interface!")

        # Reset the treatment plan label in the datahub
        Datahub.label = self.configuration['label']

        # Initialize the visualization interface
        self.visualizer = Visualizer(parent=parent)

        # Launch the visualization interface
        self.visualizer.launch()

    def compose(self):
        """Compose the treatment plan by cycling the entire workflow."""

        # Reset the treatment plan label in the datahub
        Datahub.label = self.configuration['label']

        # Cycle the workflow
        self.configure()
        self.optimize()
        self.evaluate()
        self.visualize()

    def update(
            self,
            key_value_pairs):
        """
        Update the input dictionaries by specific key-value pairs.

        Parameters
        ----------
        key_value_pairs : dict
            Dictionary with the keys and values to update.

        Raises
        ------
        KeyError
            If any update key is not included in the parameter dictionaries.
        """

        # Approve the key-value pairs
        self.input_checker.approve(key_value_pairs)

        # Loop over the items of the update dictionary
        for key, value in key_value_pairs.items():

            # Check if the key is in the configuration dictionary
            if key in self.configuration:

                # Override the configuration parameter value
                self.configuration[key] = value

            # Else, check if the key is in the optimization dictionary
            elif key in self.optimization:

                # Override the optimization parameter value
                self.optimization[key] = value

            # Else, check if the key is in the evaluation dictionary
            elif key in self.evaluation:

                # Override the evaluation parameter value
                self.evaluation[key] = value

            else:

                # Check if the logger has been initialized
                if self.logger:

                    # Log a message about the key error
                    self.logger.display_error(
                        f"The update dictionary key '{key}' is not part of "
                        "the configuration, optimization or evaluation "
                        "dictionary!")

                # Raise an error to indicate an invalid key
                raise KeyError(
                    f"The update dictionary key '{key}' is not part of the "
                    "configuration, optimization or evaluation dictionary!")
