"""Treatment plan configuration, optimization, evaluation and visualization."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from os import environ
from warnings import filterwarnings
from absl.logging import ERROR, set_verbosity

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
from pyanno4rt.evaluation import DVH
from pyanno4rt.evaluation import Dosimetrics

# Treatment plan visualization
from pyanno4rt.visualization import Visualizer

# Supporting functions
from pyanno4rt.tools import apply

# %% Set package options

set_verbosity(ERROR)
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
filterwarnings("ignore")

# %% Class definition


class TreatmentPlan():
    """
    Treatment planning class.

    This class enables configuration, optimization, outcome model building, \
    evaluation, and visualization of individual treatment plans. It therefore \
    provides a simple, but extensive interface using input dictionaries for \
    the different parameter groups.

    Parameters
    ----------
    configuration : dict
        Dictionary with the treatment plan configuration parameters:

        - ``label`` : string, unique identifier for the treatment plan;

        .. note:: Uniqueness of the label is important because it prevents \
            overwriting processes between different treatment plan instances \
            by isolating their datahubs, logging channels and storage paths.

        - ``min_log_level`` : {'debug', 'info', 'warning', 'error', \
            'critical'}, default = 'info', minimum logging level;

        - ``modality`` : {'photon', 'proton'}, treatment modality, needs to \
            be consistent with the dose calculation inputs;

        .. note:: If the modality is 'photon', a dose projection with neutral \
            RBE of 1.0 is automatically applied, whereas for the modality \
            'proton', a constant RBE of 1.1 is assumed.

        - ``number_of_fractions`` : int, number of fractions according to the \
            treatment scheme;
        - ``imaging_path`` : string, path to the CT and segmentation data;

        .. note:: It is assumed that both CT and segmentation data are \
            included in a single file (.mat or .p) or a series of files \
            (.dcm), whose content must obey the pyanno4rt data structure.

        - ``target_imaging_resolution`` : None or list, default = None, \
            imaging resolution for post-processing interpolation of the CT \
            and segmentation data, only used if a list is passed;
        - ``dose_matrix_path`` : string, path to the dose-influence matrix \
            file (.mat or .npy);
        - ``dose_resolution`` : list, size of the dose grid in [`mm`] per \
            dimension, needs to be consistent with the dose calculation inputs.

    optimization : dict
        Dictionary with the treatment plan optimization parameters:

        - ``components`` : dict, optimization components for each segmented \
            structure, i.e., objective functions and constraints;

        .. note:: The declaration scheme for a single component is

            {<segment>: [<1>, {'class': <2>, 'parameters': <3>}]}

            - <1>: 'objective' or 'constraint';
            - <2>: component label (see note below);
            - <3> parameter dictionary for the component (see the \
            component classes for details).

            Multiple objective functions or constraints can be assigned by \
            passing a list of class/parameter dictionaries.

        .. note:: The following components are currently available:

            *Objectives*

            - 'Decision Tree NTCP'
            - 'Dose Uniformity'
            - 'Equivalent Uniform Dose'
            - 'Extreme Gradient Boosting NTCP'
            - 'K-Nearest Neighbors NTCP'
            - 'Logistic Regression NTCP'
            - 'Logistic Regression TCP'
            - 'Lyman-Kutcher-Burman NTCP'
            - 'Maximum DVH'
            - 'Mean Dose'
            - 'Minimum DVH'
            - 'Moment Objective'
            - 'Naive Bayes NTCP'
            - 'Neural Network NTCP'
            - 'Neural Network TCP'
            - 'Random Forest NTCP'
            - 'Squared Deviation'
            - 'Squared Overdosing'
            - 'Squared Underdosing'
            - 'Support Vector Machine NTCP'
            - 'Support Vector Machine TCP'

            *Constraints*

            - 'Maximum Dose'
            - 'Maximum DVH'
            - 'Maximum NTCP'
            - 'Minimum Dose'
            - 'Minimum DVH'
            - 'Minimum TCP'

        - ``method`` : {'lexicographic', 'pareto', 'weighted-sum'}, default = \
            'weighted-sum', single- or multi-criteria optimization method;
        - ``solver`` : {'ipopt', 'proxmin', 'pymoo', 'pypop7', 'scipy'}, \
            default = 'scipy', python package to be used for solving the \
            optimization problem;

        .. note:: The 'ipopt' solver option requires a running IPOPT \
            installation, otherwise pyanno4rt will fall back to 'scipy'. \
            Moreover, the 'pareto' method currently only works with the \
            'pymoo' solver option.

        - ``algorithm`` : string, solution algorithm from the chosen solver:

            - ``solver`` = 'ipopt' : {'ma27', 'ma57', 'ma77', 'ma86'}, \
                default = 'ma57';
            - ``solver`` = 'proxmin' : {'admm', 'pgm', 'sdmm'}, default = \
                'pgm';
            - ``solver`` = 'pymoo' : {'NSGA3'}, default = 'NSGA3';
            - ``solver`` = 'pypop7' : {'MMES', 'LMCMA', 'RMES', 'BES', 'GS'}, \
                default = 'LMCMA';
            - ``solver`` = 'scipy' : {'L-BFGS-B', 'TNC', 'trust-constr'}, \
                default = 'L-BFGS-B'.

        .. note:: Constraints are supported by all algorithms except the \
            'L-BFGS-B' algorithm. Lower and upper variable bounds are not yet \
            supported by the 'pypop7' algorithms (but announced for the next \
            release).

        - ``initial_strategy`` : {'data-medoid', 'target-coverage', \
            'warm-start'}, default = 'target-coverage', initialization \
            strategy for the fluence vector;

        .. note:: Data-medoid initialization works best for a single dataset \
            or multiple datasets with a high degree of similarity. Otherwise, \
            the initial fluence vector may lose its individual \
            representativeness.

        - ``initial_fluence_vector`` : list or None, default = None, \
            user-defined initial fluence vector for the optimization problem, \
            only used if ``initial_strategy`` = 'warm-start';
        - ``lower_variable_bounds`` : int, float, list or None, default = 0, \
            lower bounds on the decision variables;
        - ``upper_variable_bounds`` : int, float, list or None, default = \
            None, upper bounds on the decision variables;

        .. note:: Setting lower and upper bounds for the variables can be \
            done in two ways: passing a single numeric value translates into \
            uniform bounds across all variables (where a value of None for \
            the lower and/or upper bound indicates using infinity bounds), \
            passing a list allows to set non-uniform bounds (in this case, \
            the length of the list needs to be equal to the number of \
            decision variables).

        - ``max_iter`` : int, default = 500, maximum number of iterations \
            taken for the solvers to converge;
        - ``max_cpu_time`` : float, default = 3000.0, maximum CPU time taken \
            for the solvers to converge.

    evaluation : dict, default = {}
        Dictionary with the treatment plan evaluation parameters:

        - ``dvh_type`` : {'cumulative', 'differential'}, default = \
            'cumulative', type of DVH to be calculated;
        - ``number_of_points`` : int, default = 1000, number of \
            (evenly-spaced) points for which to evaluate the DVH;
        - ``reference_volume`` : list, default = [2, 5, 50, 95, 98], \
            reference volumes for which to calculate the inverse DVH values;
        - ``reference_dose`` : list, default = [], reference dose values \
            for which to calculate the DVH values;

        .. note:: If the default value '[]' is used, reference dose levels \
            will be determined automatically in the `Dosimetrics` class.

        - ``display_segments`` : list, default = [], names of the segmented \
            structures to be displayed;

        .. note:: If the default value '[]' is used, all segments will be \
            displayed.

        - ``display_metrics`` : list, default = [], names of the plan \
            evaluation metrics to be displayed.

        .. note:: If the default value '[]' is used, all metrics will be \
            displayed.

            The following metrics are currently available: \n

            - 'mean': mean dose
            - 'std': standard deviation of the dose
            - 'max': maximum dose
            - 'min': minimum dose
            - 'Dx': dose quantile for level x
            - 'Vx': volume quantile for level x
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

    input_checker : object of class `InputChecker`
        Instance of the class 'InputChecker', which provides a unified \
        interface to perform input parameter checks in an extensible way.

    logger : object of class `Logger`
        Instance of the class `Logger`, which provides a pre-configured \
        instance of the logger together with the methods for outputting \
        messages to multiple streams.

    datahub : object of class `Datahub`
        Instance of the class `Datahub`, which provides a singleton \
        implementation for centralizing the data structures generated across \
        the program to efficiently manage and distribute information units.

    patient_loader : object of class `PatientLoader`
        Instance of the class `PatientLoader`, which provides methods for \
        importing CT and segmentation data and automatically converting them \
        to an appropriate dictionary format.

    plan_generator : object of class `PlanGenerator`
        Instance of the class `PlanGenerator`, which provides methods for \
        setting the plan properties and automatically converting them to an \
        appropriate dictionary format.

    dose_info_generator : object of class `DoseInfoGenerator`
        Instance of the class `DoseInfoGenerator`, which provides methods for \
        specifying properties related to dose (grids) and automatically \
        converting them to an appropriate dictionary format.

    fluence_optimizer : object of class `FluenceOptimizer`
        Instance of the class `FluenceOptimizer`, which provides methods for \
        preprocessing the segmentation data, setting up the optimization \
        problem and the solver, and determining the optimal fluence vector \
        and dose distribution.

    dvh : object of class `DVH`
        Instance of the class `DVH`, which provides methods to compute the \
        dose-volume histogram (DVH) as a means to evaluate the dose \
        distributions within the segments.

    dosimetrics : object of class `Dosimetrics`
        Instance of the class `Dosimetrics`, which provides methods to \
        compute dosimetrics as a means to evaluate the dose distributions \
        within the segments.

    visualizer : object of class `Visualizer`
        Instance of the class `Visualizer`, which provides classes and \
        methods to visualize different aspects related to the treatment plan.

    Examples
    --------
    Photon treatment plan on a dose grid resolution of 5mm x 5mm x 5mm, \
    30 fractions, placeholder CT/segmentation and dose-influence matrix \
    paths, three placeholder segments with associated objective functions \
    and, if applicable, placeholder features to retain with default data \
    handling & learning model parameters, as well as default optimization and \
    evaluation parameters.

    >>> tp = TreatmentPlan(
         configuration={
             'label': <your_plan_label>,
             'min_log_level': 'info',
             'modality': 'photon',
             'number_of_fractions': 30,
             'imaging_path': <your_imaging_path>,
             'target_imaging_resolution': None,
             'dose_matrix_path': <your_dose_matrix_path>,
             'dose_resolution': [5, 5, 5]
             },
         optimization={
             'components': {
                 <segment_A>: ['objective',
                               {'name': 'Dose Uniformity',
                                'parameters': {'embedding': 'active',
                                               'weight': 100,
                                               'link': None,
                                               'display'=True}}
                               ],
                 <segment_B>: ['objective',
                               {'name': 'Squared Overdosing',
                                'parameters': {'embedding': 'active',
                                               'maximum_dose': 25,
                                               'weight': 100,
                                               'link': [<segment_A>],
                                               'display'=True}}
                               ],
                 <segment_C>: ['objective',
                               {'name': 'Squared Deviation',
                                'parameters': {'embedding': 'active',
                                               'reference_dose': 60,
                                               'weight': 1000,
                                               'link': None,
                                               'display'=True}}
                               ]
                 },
             'method': 'weighted-sum',
             'solver': 'scipy',
             'algorithm': 'L-BFGS-B',
             'initial_strategy': 'target-coverage',
             'initial_fluence_vector': None,
             'lower_variable_bounds': 0,
             'upper_variable_bounds': None,
             'max_iter': 500,
             'max_cpu_time': 3000.0
             },
         evaluation={
             'dvh_type': 'cumulative',
             'number_of_points': 1000,
             'reference_volume': [2, 5, 50, 95, 98],
             'reference_dose': [],
             'display_segments': [],
             'display_metrics': []
             }
         )
    >>> tp.configure()
    >>> tp.optimize()
    >>> tp.evaluate()
    >>> tp.visualize()

    Alternatively, the four method calls at the bottom can be abbreviated by \
    the shortcut:

    >>> tp.compose()

    To change the input values of the class, you can call an instance method:

    >>> tp.update(key_value_pairs)

    where the items in ``key_value_pairs`` indicate which key from the \
    input dictionaries should be updated by which value.

    Saving a treatment plan can be achieved with a snapshot:

    >>> from pyanno4rt.tools import snapshot
    >>> snapshot(tp, <your_plan_path>)

    Loading a treatment plan is possible with a copycat:

    >>> from pyanno4rt.tools import copycat
    >>> tp = copycat(TreatmentPlan, <your_plan_path>)

    Finally, the graphical user interface with the treatment plan can be \
    launched via:

    >>> from pyanno4rt.gui import GraphicalUserInterface
    >>> gui = GraphicalUserInterface()
    >>> gui.launch(tp)

    More detailed setups of `TreatmentPlan` instances, including machine \
    learning model-based components, can be found in the head-and-neck \
    example folder (downloadable from our Github repository under \
    https://github.com/pyanno4rt/pyanno4rt).
    """

    def __init__(
            self,
            configuration,
            optimization,
            evaluation=None):

        # Initialize the (optional) evaluation dictionary
        evaluation = evaluation if evaluation is not None else {}

        # Initialize the input checker
        self.input_checker = InputChecker()

        # Approve the input types
        self.input_checker.approve({'configuration': configuration,
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
                'ma57' if optimization.get('solver') == 'ipopt'
                else 'pgm' if optimization.get('solver') == 'proxmin'
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
            'max_cpu_time': optimization.get('max_cpu_time', 3000.0)
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

        # Approve the dictionary values
        apply(self.input_checker.approve,
              (self.configuration, self.optimization, self.evaluation))

        # Initialize the instance attributes
        self.logger = None
        self.datahub = None
        self.patient_loader = None
        self.plan_generator = None
        self.dose_info_generator = None
        self.fluence_optimizer = None
        self.histogram = None
        self.dosimetrics = None
        self.visualizer = None

    def configure(self):
        """Initialize the configuration classes and process the input data."""

        # Initialize the logger
        self.logger = Logger(
            self.configuration['label'], self.configuration['min_log_level'])

        # Initialize the datahub
        self.datahub = Datahub(
            self.configuration['label'], self.logger, self.input_checker)

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
            dose_resolution=self.configuration['dose_resolution'],
            number_of_fractions=self.configuration['number_of_fractions'],
            dose_path=self.configuration['dose_matrix_path'])

        # Load the patient data
        self.patient_loader.load()

        # Generate the plan information
        self.plan_generator.generate()

        # Generate the dose information
        self.dose_info_generator.generate()

    def optimize(self):
        """Initialize the fluence optimizer and solve the problem."""

        # Check if any required attribute is missing
        if any(getattr(self, attribute) is None for attribute in (
                'logger', 'datahub', 'input_checker', 'patient_loader',
                'plan_generator', 'dose_info_generator')):

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
            max_cpu_time=self.optimization['max_cpu_time'])

        # Solve the optimization problem with the optimizer
        self.fluence_optimizer.solve()

    def evaluate(self):
        """Initialize the evaluation classes and compute the plan metrics."""

        # Check if the 'fluence_optimizer' attribute is missing
        if getattr(self, 'fluence_optimizer') is None:

            # Raise an error to indicate the missing attribute
            raise AttributeError(
                "Please optimize the treatment plan before evaluation!")

        # Reset the treatment plan label in the datahub
        Datahub.label = self.configuration['label']

        # Initialize the DVH class
        self.histogram = DVH(
            dvh_type=self.evaluation['dvh_type'],
            number_of_points=self.evaluation['number_of_points'],
            display_segments=self.evaluation['display_segments'])

        # Initialize the dosimetrics class
        self.dosimetrics = Dosimetrics(
            reference_volume=self.evaluation['reference_volume'],
            reference_dose=self.evaluation['reference_dose'],
            display_segments=self.evaluation['display_segments'],
            display_metrics=self.evaluation['display_metrics'])

        # Compute the dose-volume histogram from the optimized dose
        self.histogram.compute(self.datahub.optimization['optimized_dose'])

        # Compute the dosimetrics from the optimized dose
        self.dosimetrics.compute(self.datahub.optimization['optimized_dose'])

    def visualize(self, parent=None):
        """Initialize the visualization interface and launch it."""

        # Check if any required attribute is missing
        if any(getattr(self, attribute) is None for attribute in (
                'fluence_optimizer', 'histogram', 'dosimetrics')):

            # Raise an error to indicate a missing attribute
            raise AttributeError(
                "Please optimize and evaluate the treatment plan before "
                "launching the visualization interface!")

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
            Dictionary with the values to update.
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

                # Raise an error to indicate an invalid key
                raise KeyError(
                    f"The update dictionary key '{key}' is not part of the "
                    "configuration, optimization or evaluation dictionary!")
