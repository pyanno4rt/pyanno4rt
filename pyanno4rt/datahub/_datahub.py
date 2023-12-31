"""Central data storage and management hub."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% Class definition


class Datahub():
    """
    Central data storage and management hub class.

    This class provides a singleton implementation for centralizing the data \
    structures generated across the pyanno4rt program to efficiently manage \
    and distribute information units, e.g. CT and segmentation data or \
    optimization and outcome model results.

    Parameters
    ----------
    args : tuple
        Tuple with optional (non-keyworded) parameters. The value ``args[0]``
        refers to the label of the treatment plan, while ``args[1]`` \
        represents an instance of `Logger` and ``args[2]`` an instance of \
        `InputChecker`. Only required for (re-)instantiating a datahub.

    Attributes
    ----------
    instances : dict
        Dictionary with pairs of treatment plan labels and associated \
        `Datahub` instances. For each unique treatment plan label, a new \
        instance of `Datahub` is generated, which can then be used across the \
        plan instance. This dictionary should NEVER be changed manually!

    label : string
        Label of the current active´ treatment plan instance.

    input_checker : object of class `InputChecker`
        Instance of the class 'InputChecker', which provides a unified \
        interface to perform input parameter checks in an extensible way.

    logger : object of class `Logger`
        Instance of the class `Logger`, which provides a pre-configured \
        instance of the logger together with the methods for outputting \
        messages to multiple streams.

    computed_tomography : dict
        Dictionary with information on the CT images.

    segmentation : dict
        Dictionary with information on the segmented structures.

    plan_configuration : dict
        Dictionary with information on the plan configuration.

    dose_information : dict
        Dictionary with information on the dose grid.

    optimization : dict
        Dictionary with information on the fluence optimization.

    datasets : dict
        Dictionary with pairs of model labels and associated external \
        datasets used for model fitting. Each dataset is a dictionary itself, \
        holding information on the raw data and the features/labels.

    feature_maps : dict
        Dictionary with pairs of model labels and associated feature maps. \
        Each feature map holds links between the features from the respective \
        dataset, the segments, and the definitions from the feature catalogue.

    model_instances : dict
        Dictionary with pairs of model labels and associated model instances, \
        i.e., the prediction model, the model configuration dictionary, and \
        the model hyperparameters obtained from hyperparameter tuning.

    model_inspections : dict
        Dictionary with pairs of model labels and associated model \
        inspectors. Each inspector holds information on the inspection \
        measures calculated.

    model_evaluations : dict
        Dictionary with pairs of model labels and associated model \
        evaluators. Each evaluator holds information on the evaluation \
        measures calculated.

    histogram : dict
        Dictionary with information on the cumulative or differential \
        dose-volume histograms for each segmented structure.

    dosimetrics : dict
        Dictionary with information on the dosimetrics for each segmented \
        structure.
    """

    # Initialize the datahub instances dictionary
    instances = {}

    # Initialize the information units
    label = None
    input_checker = None
    logger = None
    object_stream = None
    computed_tomography = None
    segmentation = None
    plan_configuration = None
    dose_information = None
    optimization = {}
    datasets = {}
    feature_maps = {}
    model_instances = {}
    model_inspections = {}
    model_evaluations = {}
    histogram = None
    dosimetrics = None

    def __new__(
            cls,
            *args):
        """
        Create a new object of the class.

        Parameters
        ----------
        args : tuple
            Tuple with optional (non-keyworded) datahub parameters. The value \
            ``args[0]`` refers to the label of the treatment plan, while \
            ``args[1]`` represents an instance of `Logger` and ``args[2]`` an \
            instance of `InputChecker`.

        Returns
        -------
        object of class `Datahub`
            Instance of the class `Datahub`, which provides a singleton \
            implementation for centralizing the data structures generated \
            across the pyanno4rt program to efficiently manage and distribute \
            information units.

        Notes
        -----
        The ``args`` tuple is passed during the datahub initialization in the \
        treatment plan class only, i.e., the generated instance of `Datahub` \
        is appended to the ``instances`` dictionary once, and otherwise \
        fetched from the it. If ``args`` is None, and the instance for the \
        treatment plan label does not exist, a default object is returned \
        without appending to the ``instances`` dictionary.
        """

        # Check if no arguments are passed
        if not args:

            # Check if the instance already exists in the dictionary
            if cls.instances.get(cls.label):

                # Get the instance from the dictionary
                instance = cls.instances[cls.label]

            else:

                # Create a new default instance
                instance = super().__new__(cls)

        else:

            # Check if the instance does not already exist in the dictionary
            if not cls.instances.get(args[0]):

                # Create a new instance and append it to the dictionary
                cls.instances[args[0]] = super().__new__(cls)

            # Get the instance from the dictionary
            instance = cls.instances[args[0]]

        return instance

    def __init__(
            self,
            *args):

        # Check if arguments are passed
        if args:

            # Set the instance and the class label to the same value
            self.label = args[0]
            Datahub.label = args[0]

            # Set the logging object
            self.logger = args[1]

            # Set the input checking object
            self.input_checker = args[2]

            # Log a message about the initialization of the class
            self.logger.display_info("Initializing datahub ...")
