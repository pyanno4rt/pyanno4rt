"""Central data storage and management hub."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% Class definition


class Datahub():
    """
    Central data storage and management hub class.

    This class provides a singleton datahub for centralizing the information \
    units generated across one or multiple treatment plans, e.g. dictionaries \
    with CT and segmentation data, to efficiently manage and distribute them.

    Parameters
    ----------
    *args : tuple
        Tuple with optional (non-keyworded) parameters. The element args[0]
        refers to the treatment plan label, while args[1] is a \
        :class:`~pyanno4rt.logging._logger.Logger` object and args[2] \
        is an :class:`~pyanno4rt.input_check._input_checker.InputChecker` \
        object. Only required for (re-)instantiating a datahub.

    Attributes
    ----------
    instances : dict
        Dictionary with pairs of treatment plan labels and associated \
        :class:`~pyanno4rt.datahub._datahub.Datahub` objects.

    label : None or str
        Label of the current active treatment plan instance.

    input_checker : None or object of class \
        :class:`~pyanno4rt.input_check._input_checker.InputChecker`
        The object used to approve the input dictionaries.

    logger : None or object of class \
        :class:`~pyanno4rt.logging._logger.Logger`
        The object used to print and store logging messages.

    computed_tomography : None or dict
        Dictionary with information on the CT images.

    segmentation : None or dict
        Dictionary with information on the segmented structures.

    plan_configuration : None or dict
        Dictionary with information on the plan configuration.

    dose_information : None or dict
        Dictionary with information on the dose grid.

    optimization : None or dict
        Dictionary with information on the fluence optimization.

    datasets : None or dict
        Dictionary with pairs of model labels and associated external \
        datasets used for model fitting. Each dataset is a dictionary itself, \
        holding information on the raw data and the features/labels.

    feature_maps : None or dict
        Dictionary with pairs of model labels and associated feature maps. \
        Each feature map holds links between the features from the respective \
        dataset, the segments, and the definitions from the feature catalogue.

    model_instances : None or dict
        Dictionary with pairs of model labels and associated model instances, \
        i.e., the prediction model, the model configuration dictionary, and \
        the model hyperparameters obtained from hyperparameter tuning.

    model_inspections : None or dict
        Dictionary with pairs of model labels and associated model \
        inspectors. Each inspector holds information on the inspection \
        measures calculated.

    model_evaluations : None or dict
        Dictionary with pairs of model labels and associated model \
        evaluators. Each evaluator holds information on the evaluation \
        measures calculated.

    dose_histogram : None or dict
        Dictionary with information on the cumulative or differential \
        dose-volume histogram for each segmented structure.

    dosimetrics : None or dict
        Dictionary with information on the dosimetrics for each segmented \
        structure.
    """

    # Initialize the datahub instances dictionary
    instances = {}

    # Initialize the information units
    label = None
    input_checker = None
    logger = None
    computed_tomography = None
    segmentation = None
    plan_configuration = None
    dose_information = None
    optimization = None
    datasets = None
    feature_maps = None
    model_instances = None
    model_inspections = None
    model_evaluations = None
    dose_histogram = None
    dosimetrics = None

    def __new__(
            cls,
            *args):
        """
        Create a new object of the class.

        Parameters
        ----------
        *args : tuple
            See the class docstring.

        Returns
        -------
        object of class :class:`~pyanno4rt.datahub._datahub.Datahub`
        """

        # Check if no arguments are passed
        if not args:

            # Check if the instance already exists in the dictionary
            if cls.instances.get(cls.label):

                # Return the instance from the dictionary
                return cls.instances[cls.label]

            # Return a new default instance
            return super().__new__(cls)

        # Check if the instance does not already exist in the dictionary
        if not cls.instances.get(args[0]):

            # Create a new instance and append it to the dictionary
            cls.instances[args[0]] = super().__new__(cls)

        # Return the instance from the dictionary
        return cls.instances[args[0]]

    def __init__(
            self,
            *args):

        # Check if arguments are passed
        if args:

            # Set the instance and the class label to the same value
            self.label = Datahub.label = args[0]

            # Set the input checking object
            self.input_checker = args[1]

            # Set the logging object
            self.logger = args[2]

            # Log a message about the initialization of the class
            self.logger.display_info("Initializing datahub ...")
