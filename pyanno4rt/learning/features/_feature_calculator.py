"""Feature calculator."""

# Author: Tim Ortkamp

# %% External package import

from numpy import (
    array, array_equal, fromiter, pad, unravel_index, vstack, zeros)

# %% Internal package import

from pyanno4rt.logging import get_logger

# %% Class definition


class FeatureCalculator():
    """
    Feature calculator class.

    This class implements methods to (re)calculate input features and their \
    dose gradients for a specific feature-to-function mapping.

    Parameters
    ----------
    handlers : dict
        Dictionary with the handlers (patient, plan, dose).

    Attributes
    ----------
    handlers : dict
        See 'Parameters'.

    radiomics : None or dict
        Dictionary with the (cached) radiomic feature names and values.

    statics : None or dict
        Dictionary with the (cached) static feature names and values.

    feature_map : None or dict
        Dictionary with mappings between features and calculation functions.

    feature_history : None or list
        Feature values per iteration.

    inputs : None or dict
        Dictionary with the input values for feature calculation.

    _iteration : list
        Iteration numbers for synchronization with the optimization problem.

    _dose_cache : tuple
        Cache for the dose values.

    _feature_cache : ndarray
        Cache for the feature values.
    """

    def __init__(
            self,
            handlers):

        # Get the handlers
        self.handlers = handlers

        # Initialize the radiomics and statics dictionaries
        self.radiomics = None
        self.statics = None

        # Initialize the feature map and history
        self.feature_map = None
        self.feature_history = None

        # Initialize the input dictionary
        self.inputs = None

        # Initialize the iteration numbers for synchronization
        self._iteration = [0, 0]

        # Initialize the dose and feature cache
        self._dose_cache = tuple(array([]))
        self._feature_cache = array([])

    def set_mapping(
            self,
            feature_map,
            return_self=False,
            verbose=True):
        """
        Set the feature-to-function mapping.

        Parameters
        ----------
        feature_map : dict
            Dictionary with mappings between features and calculation \
            functions.

        return_self : bool, default=False
            Indicator for returning the instance.

        verbose : bool, default=True
            Indicator for logging output messages.

        Returns
        -------
        None or object of class \
            :class:`~pyanno4rt.learning.features._feature_calculator.FeatureCalculator`
            None if `return_self` is False, else the instance.
        """

        # Check if messages should be printed
        if verbose:

            # Log a message about setting the feature map
            get_logger().info("Setting feature map for (re)calculation ...")

        # Initialize the radiomics and statics dictionaries
        self.radiomics = {}
        self.statics = {}

        # Initialize the feature map
        self.feature_map = feature_map

        # Initialize the feature history
        self.feature_history = []

        # Initialize the input dictionary
        self.inputs = {
            'dose': None,
            'dose_cube': None,
            'indices': None,
            'paddings': None,
            'masks': None}

        # Check if the instance should be returned
        if return_self:

            # Return the instance
            return self

        # Else, return None
        return None

    def precompute(
            self,
            dose,
            segment):
        """
        Precompute the input quantities.

        Parameters
        ----------
        dose : tuple
            Dose vectors.

        segment : tuple
            Segment names.
        """

        def get_dose(dose):
            """Get the dose vectors."""

            return (
                dose_handler.number_of_fractions * subdose.ravel()
                for subdose in dose)

        def get_dose_cubes(segment):
            """Get the dose cubes for all segments."""

            def subcube(subsegment):
                """Get the dose cube for a single segment."""

                # Initialize the dose cube
                dose_cube = zeros(dimensions)

                # Insert the dose values of the segment
                dose_cube[unravel_index(
                    segmentation[subsegment]['resized_indices'], dimensions,
                    order='F')] = self.inputs['dose'][subsegment]

                return dose_cube

            # Get the dose cube dimensions
            dimensions = dose_handler.cube_dimensions

            return map(subcube, segment)

        def get_indices(segment):
            """Get the segment indices."""

            return (
                segmentation[subsegment]['resized_indices']
                for subsegment in segment)

        def get_paddings():
            """Get the gradient paddings."""

            # Get the lengths of the segment indices
            lengths = tuple(map(len, self.inputs['indices'].values()))

            return (tuple(
                (0, sum(lengths[1:]))
                if index == 0
                else (sum(lengths[:index]), sum(lengths[index+1:]))
                if index < len(lengths)-1
                else (sum(lengths[:index]), 0)
                for index, _ in enumerate(lengths))
                + ((0, sum(lengths)),))

        def get_masks(segment):
            """Get the segment masks."""

            def submasks(subsegment):
                """Get the masks for a single segment."""

                # Initialize the masks
                radiomics_mask = zeros(ct_dimensions)
                dose_mask = zeros(dose_dimensions)

                # Insert ones at the CT indices of the segment
                radiomics_mask[unravel_index(
                    segmentation[subsegment]['raw_indices'], ct_dimensions,
                    order='F')] = 1

                # Insert ones at the dose indices of the segment
                dose_mask[unravel_index(
                    segmentation[subsegment]['resized_indices'],
                    dose_dimensions, order='F')] = 1

                return (radiomics_mask, dose_mask)

            # Get the CT and dose cube dimensions
            ct_dimensions = computed_tomography['cube_dimensions']
            dose_dimensions = dose_handler.cube_dimensions

            return map(submasks, segment)

        # Get the information units
        computed_tomography = (
            self.handlers['patient_handler'].computed_tomography)
        segmentation = self.handlers['patient_handler'].segmentation
        dose_handler = self.handlers['dose_handler']

        # Add the precomputed dose vectors
        self.inputs['dose'] = dict(zip(segment, get_dose(dose)))

        # Add the precomputed dose cubes
        self.inputs['dose_cube'] = dict(zip(segment, get_dose_cubes(segment)))

        # Check if the segment indices have not been computed yet
        if self.inputs['indices'] is None:

            # Add the precomputed segment indices
            self.inputs['indices'] = dict(zip(segment, get_indices(segment)))

            # Add the precomputed gradient paddings
            self.inputs['paddings'] = dict(zip(
                segment+(None,), get_paddings()))

        # Check if the segment masks have not been computed yet
        if (self.inputs['masks'] is None
                or tuple(segment) != (*self.inputs['masks'],)):

            # Add the precomputed masks by segment
            self.inputs['masks'] = dict(zip(segment, get_masks(segment)))

    def featurize(
            self,
            dose,
            segment,
            update_cache=False):
        """
        Convert dose and segment information into the feature vector.

        Parameters
        ----------
        dose : tuple
            Dose vectors.

        segment : tuple
            Segment names.

        update_cache : bool, default=False
            Indicator for enforcing cache updates.

        Returns
        -------
        ndarray
            Feature vector.
        """

        # Check if the caches should be updated
        if (len(self._feature_cache) == 0
                or self._iteration[0] != self._iteration[1]
                or update_cache):

            # Synchronize the iteration numbers
            self._iteration[0] = self._iteration[1]

            # Check if the dose has changed
            if not array_equal(dose, self._dose_cache):

                # Precompute the input quantities
                self.precompute(dose, segment)

                # Update the dose cache
                self._dose_cache = dose

            # Compute and cache the feature vector
            self._feature_cache = self._compute_features()

        return self._feature_cache

    def _compute_features(self):
        """
        Compute the feature vector.

        Returns
        -------
        ndarray
            Feature vector.
        """

        def compute_feature(feature):
            """Compute a single feature value."""

            def get_dosiomic():
                """Get the value for a dosiomic feature."""

                return mapping['computation'](
                    self.inputs['dose'][mapping['segment']],
                    self.inputs['dose_cube'][mapping['segment']],
                    fromiter(dose_resolution.values(), float),
                    self.inputs['masks'][mapping['segment']][1])

            def get_radiomic():
                """Get the value for a radiomic feature."""

                # Check if the feature has not been computed yet
                if feature not in self.radiomics:

                    # Add the feature value to the radiomics dictionary
                    self.radiomics[feature] = mapping['computation'](
                            self.inputs['masks'][mapping['segment']][0],
                            fromiter(
                                computed_tomography['resolution'].values(),
                                float))

                return self.radiomics[feature]

            def get_static():
                """Get the value for a static feature."""

                # Check if the feature has not been included yet
                if feature not in self.statics:

                    # Add the feature value to the statics dictionary
                    self.statics[feature] = mapping['value']

                return self.statics[feature]

            # Map the feature classes to the calculation functions
            functions = {
                'Dosiomics': get_dosiomic,
                'Radiomics': get_radiomic,
                'Statics': get_static}

            # Get the feature-to-function mapping
            mapping = self.feature_map[feature]

            # Compute the feature value
            feature_value = functions[mapping['class']]()

            return feature_value

        # Get the CT and dose resolution
        computed_tomography = (
            self.handlers['patient_handler'].computed_tomography)
        dose_resolution = self.handlers['dose_handler'].resolution

        # Compute the features from the mapping
        feature_vector = array(tuple(map(compute_feature, self.feature_map)))

        # Add the feature vector to the history
        self.feature_history.append(feature_vector)

        return feature_vector.reshape(1, -1)

    def gradientize(
            self,
            dose,
            segment):
        """
        Convert dose and segment information into the dose Jacobian.

        Parameters
        ----------
        dose : tuple
            Dose vectors.

        segment : tuple
            Segment names.

        Returns
        -------
        ndarray
            Dose Jacobian.
        """

        def compute_gradient(feature):
            """Compute a single gradient."""

            def get_dosiomic():
                """Get the gradient for a dosiomic feature."""

                # Calculate the dose gradient
                dose_gradient = mapping['differentiation'](
                    self.inputs['dose'][mapping['segment']],
                    self.inputs['dose_cube'][mapping['segment']],
                    fromiter(dose_resolution.values(), float),
                    self.inputs['masks'][mapping['segment']][1])

                # Check if the feature is cube-based
                if mapping['name'] in (
                        'Dose Gradient', 'Dose Moment', 'Dose Subvolume'):

                    # Return the indexed dose gradient
                    return dose_gradient[
                        self.inputs['indices'][mapping['segment']]]

                return dose_gradient

            def get_radiomic():
                """Get the gradient for a radiomic feature."""

                return zeros(
                    (len(self.inputs['indices'][mapping['segment']]),))

            def get_static():
                """Get the gradient for a static feature."""

                return zeros((0,))

            # Map the feature classes to the calculation functions
            functions = {
                'Dosiomics': get_dosiomic,
                'Radiomics': get_radiomic,
                'Statics': get_static}

            # Get the feature-to-function mapping
            mapping = self.feature_map[feature]

            # Compute the feature gradient
            feature_gradient = functions[mapping['class']]()

            return pad(
                feature_gradient,
                self.inputs['paddings'][mapping.get('segment')])

        # Get the dose resolution
        dose_resolution = self.handlers['dose_handler'].resolution

        # Check if the dose has changed
        if not array_equal(dose, self._dose_cache):

            # Precompute the input quantities
            self.precompute(dose, segment)

            # Update the dose cache
            self._dose_cache = dose

        # Compute the gradients from the mapping
        gradients = map(compute_gradient, self.feature_map)

        return vstack(tuple(gradients)).T

    def history_to_dict(self):
        """Convert the feature history to a dictionary."""

        self.feature_history = dict(zip(
            (*self.feature_map,),
            (*vstack(self.feature_history)[1:, :].transpose(),)))
