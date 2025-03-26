"""Feature values and gradient (re)calculation."""

# Author: Tim Ortkamp

# %% External package import

from numpy import (
    array, array_equal, empty, fromiter, pad, unravel_index, vstack, zeros)
from scipy.ndimage import zoom

# %% Internal package import

from pyanno4rt.datahub import Datahub

# %% Class definition


class FeatureCalculator():
    """
    Feature values and gradient (re)calculation class.

    Parameters
    ----------
    write_features : bool
        Indicator for tracking the feature values.

    verbose : bool, default=True
        Boolean indicator for the logging of the initialization.

    Attributes
    ----------
    write_features : bool
        See 'Parameters'.

    radiomics : dict
        Dictionary with the radiomic feature names and values for caching.

    statics : dict
        Dictionary with the static (fixed) feature names and values.

    feature_map : dict
        Dictionary with the mappings of feature names, segments and \
        computation/differentiation functions.

    feature_history : ndarray or None
        Feature values per iteration (only if ``write_features`` is True).

    inputs : dict
        Dictionary with the input values for feature calculation.

    __iteration__ : list
        Iteration numbers for feature calculation and optimization problem.

    __dose_cache__ : tuple
        Cache tuple for the dose values.

    __feature_cache__ : ndarray
        Cache array for the feature values.
    """

    def __init__(
            self,
            write_features,
            verbose=True):

        # Check if verbose is True
        if verbose:

            # Log a message about the initialization of the class
            Datahub().logger.display_info(
                "Initializing feature calculator ...")

        # Get the feature writing indicator from the argument
        self.write_features = write_features

        # Initialize the radiomic and static feature dictionaries
        self.radiomics = {}
        self.statics = {}

        # Initialize the feature map and history
        self.feature_map = None
        self.feature_history = None

        # Initialize the input dictionary
        self.inputs = {
            'dose': None,
            'dose_cube': None,
            'indices': None,
            'paddings': None,
            'require_cube': ('doseSubvolume', 'doseMoment', 'doseGradient'),
            'require_spacing': ('doseGradient',),
            'masks': None}

        # Initialize the iteration numbers for synchronization
        self.__iteration__ = [0, 0]

        # Initialize the dose and the feature cache
        self.__dose_cache__ = tuple(array([]))
        self.__feature_cache__ = array([])

    def add_static_map(
            self,
            statics):
        """."""

        # Log a message about the static values map addition
        Datahub().logger.display_info(
            "Adding static values map to the feature calculator ...")

        # Initialize the static values map from the argument
        self.statics = statics

    def add_feature_map(
            self,
            feature_map,
            return_self=False):
        """
        Add the feature map to the calculator.

        Parameters
        ----------
        feature_map : dict
            ...
        """

        # Log a message about the feature map addition
        Datahub().logger.display_info(
            "Adding feature map to the feature calculator ...")

        # Initialize the feature map from the argument
        self.feature_map = feature_map

        # Check if the feature values should be stored in a history
        if self.write_features:

            # Initialize the feature history from the argument
            self.feature_history = empty(shape=(1, len(self.feature_map)))

        # Check if the instance should be returned
        if return_self:

            return self

        return None

    def precompute(
            self,
            dose,
            segment):
        """
        Precompute the input quantities for the feature calculation.

        Parameters
        ----------
        dose : tuple
            Tuple with the dose values.

        segment : tuple
            Tuple with the segment names.
        """

        # Initialize the datahub
        hub = Datahub()

        # Get the information units from the datahub
        segmentation = hub.segmentation
        dose_information = hub.dose_information

        def precompute_dose(dose):
            """Precompute the dose."""

            return (
                dose_information['number_of_fractions'] * subdose.ravel()
                for subdose in dose)

        def precompute_dose_cube(segment):
            """Precompute the dose cube."""

            # Get the dose grid dimensions
            dimensions = dose_information['cube_dimensions']

            def get_subsegment_cube(subsegment):
                """Get the dose cube for a single segment."""

                # Initialize the dose cube
                dose_cube = zeros(dimensions)

                # Insert the dose values of the segment into the dose cube
                dose_cube[unravel_index(
                    segmentation[subsegment]['resized_indices'], dimensions,
                    order='F')] = self.inputs['dose'][subsegment]

                return dose_cube

            return (get_subsegment_cube(subsegment) for subsegment in segment)

        def precompute_indices(segment):
            """Precompute the segment indices."""

            return (
                segmentation[subsegment]['resized_indices']
                for subsegment in segment)

        def precompute_paddings():
            """Precompute the gradient paddings."""

            # Get the lengths of the segment indices
            lengths = tuple(
                len(index) for index in self.inputs['indices'].values())

            return (tuple(
                (0, sum(lengths[1:]))
                if index == 0
                else (sum(lengths[:index]), sum(lengths[index+1:]))
                if index < len(lengths)-1
                else (sum(lengths[:index]), 0)
                for index, _ in enumerate(lengths))
                + ((0, sum(lengths)),))

        def precompute_masks(segment):
            """Precompute the segment masks."""

            # Get the cube dimensions from the information unit
            ct_dimensions = hub.computed_tomography['cube_dimensions']
            dose_dimensions = dose_information['cube_dimensions']

            def get_subsegment_masks(subsegment):
                """Get the masks for a single segment."""

                # Initialize the radiomics mask
                radiomics_mask = zeros(ct_dimensions)

                # Insert ones at the indices of the segment
                radiomics_mask[unravel_index(
                    segmentation[subsegment]['raw_indices'],
                    ct_dimensions, order='F')] = 1

                # Get the dose mask
                dose_mask = zoom(
                    radiomics_mask, (pair[0]/pair[1] for pair in zip(
                        dose_dimensions, ct_dimensions)), order=0)

                return (radiomics_mask, dose_mask)

            return (get_subsegment_masks(subsegment) for subsegment in segment)

        # Add the precomputed dose vectors to the input dictionary
        self.inputs['dose'] = dict(
            zip(segment, precompute_dose(dose)))

        # Add the precomputed dose cubes to the input dictionary
        self.inputs['dose_cube'] = dict(
            zip(segment, precompute_dose_cube(segment)))

        # Check if the segment indices have not been computed yet
        if self.inputs['indices'] is None:

            # Add the precomputed segment indices to the input dictionary
            self.inputs['indices'] = dict(
                zip(segment, precompute_indices(segment)))

            # Add the precomputed gradient paddings to the input dictionary
            self.inputs['paddings'] = dict(
                zip(segment+[None], precompute_paddings()))

        # Check if the segment masks have not been computed yet
        if (self.inputs['masks'] is None
                or tuple(segment) != (*self.inputs['masks'],)):

            # Add the precomputed masks by segment to the input dictionary
            self.inputs['masks'] = dict(
                zip(segment, precompute_masks(segment)))

    def featurize(
            self,
            dose,
            segment,
            no_cache=False):
        """
        Transform dose and segment information into the feature vector.

        Parameters
        ----------
        dose : tuple
            Tuple with the dose values.

        segment : tuple
            Tuple with the segment names.

        Returns
        -------
        ndarray
            Feature vector.
        """

        # Check if the feature cache needs to be updated
        if (len(self.__feature_cache__) == 0
                or self.__iteration__[0] != self.__iteration__[1]
                or no_cache):

            # Synchronize the iteration numbers
            self.__iteration__[0] = self.__iteration__[1]

            # Check if the dose has changed
            if not array_equal(dose, self.__dose_cache__):

                # Precompute the input for the feature calculation
                self.precompute(dose, segment)

                # Update the dose cache
                self.__dose_cache__ = dose

            # Retrieve and cache the feature vector
            self.__feature_cache__ = self.get_feature_vector()

        return self.__feature_cache__

    def get_feature_vector(self):
        """
        Get the feature vector.

        Returns
        -------
        ndarray
            Feature vector.
        """

        # Initialize the datahub
        hub = Datahub()

        def compute_feature_value(feature):
            """Compute a single feature value."""

            def get_dosiomic_value(feature, segment):
                """Get the value of a dosiomic feature."""

                # Determine the number of input conditions fulfilled
                boolean_sum = sum((
                    any(label in feature
                        for label in self.inputs['require_cube']),
                    any(label in feature
                        for label in self.inputs['require_spacing'])))

                # Check if the boolean sum is zero
                if boolean_sum == 0:

                    # Compute the feature value
                    return self.feature_map[feature]['computation'](
                        self.inputs['dose'][segment])

                # Check if the boolean sum is one
                if boolean_sum == 1:

                    # Compute the feature value
                    return self.feature_map[feature]['computation'](
                        self.inputs['dose'][segment],
                        self.inputs['dose_cube'][segment])

                # Else, compute the feature value for the boolean sum of two
                return self.feature_map[feature]['computation'](
                    self.inputs['dose'][segment],
                    self.inputs['dose_cube'][segment],
                    fromiter(
                        hub.dose_information['resolution'].values(), float),
                    self.inputs['masks'][segment][1])

            def get_radiomic_value(feature, segment):
                """Get the value of a radiomic feature."""

                # Check if the feature has already been computed
                if feature in self.radiomics:

                    # Return the feature value from the radiomics dictionary
                    return self.radiomics[feature]

                # Compute the feature value
                self.radiomics[feature] = self.feature_map[
                    feature]['computation'](
                        self.inputs['masks'][segment][0],
                        fromiter(
                            hub.computed_tomography['resolution'].values(),
                            float))

                return self.radiomics[feature]

            def get_static_value(feature, _):
                """Get the value of a static feature."""

                # Check if the feature is included as static
                if feature in self.statics:

                    # Return the value from the static feature dictionary
                    return self.statics[feature]

                # Log a message about a missing static feature
                hub.logger.display_error(
                    f"The feature '{feature}' is missing in the static "
                    "values map ...")

                # Raise an attribute error
                raise AttributeError

            # Map the feature types to the get functions
            get_functions = {
                'Dosiomics': get_dosiomic_value,
                'Radiomics': get_radiomic_value,
                'Statics': get_static_value}

            # Run the specific get function to retrieve the feature value
            feature_value = get_functions[self.feature_map[feature]['class']](
                feature, self.feature_map[feature]['segment'])

            return feature_value

        # Run the computation function for all features in the feature map
        features = map(compute_feature_value, (*self.feature_map,))

        # Convert the features into a shaped array
        feature_vector = array((*features,)).reshape(1, -1)

        # Check if the feature history should be written
        if self.write_features and self.__iteration__[1] >= 2:

            # Add the feature vector to the history
            self.feature_history = vstack((
                self.feature_history, feature_vector))

        return feature_vector

    def gradientize(
            self,
            dose,
            segment):
        """
        Transform dose and segment information into the gradient matrix.

        Parameters
        ----------
        dose : tuple
            Tuple with the dose values.

        segment : tuple
            Tuple with the segment names.

        Returns
        -------
        csr_matrix
            Gradient matrix.
        """

        # Initialize the datahub
        hub = Datahub()

        # Get the information units from the datahub
        dose_information = hub.dose_information

        # Check if the dose has changed
        if not array_equal(dose, self.__dose_cache__):

            # Precompute the input for the feature calculation
            self.precompute(dose, segment)

            # Update the dose cache
            self.__dose_cache__ = dose

        def compute_feature_gradient(feature):
            """Compute a single gradient."""

            def get_dosiomic_gradient(feature, segment):
                """Get the gradient of a dosiomic feature."""

                # Get the inputs
                dose = self.inputs['dose'][segment]
                dose_cube = self.inputs['dose_cube'][segment]
                indices = self.inputs['indices'][segment]
                require_cube = self.inputs['require_cube']
                require_spacing = self.inputs['require_spacing']
                masks = self.inputs['masks'][segment]

                # Get the gradient function
                differentiate = self.feature_map[feature]['differentiation']

                # Determine the number of input conditions fulfilled
                boolean_sum = sum((
                    any(label in feature for label in require_cube),
                    any(label in feature for label in require_spacing)))

                # Check if the boolean sum is zero
                if boolean_sum == 0:

                    # Compute the gradient vector
                    return differentiate(
                        dose, dose_information['number_of_voxels'], indices)

                # Check if the boolean sum is one
                if boolean_sum == 1:

                    # Compute the gradient vector
                    return differentiate(dose, dose_cube)[indices]

                # Else, compute the gradient vector for the boolean sum of two
                return differentiate(
                    dose, dose_cube,
                    fromiter(dose_information['resolution'].values(), float),
                    masks[1])[indices]

            def get_radiomic_gradient(_, segment):
                """Get the gradient of a radiomic feature."""

                return zeros((len(self.inputs['indices'][segment]),))

            def get_static_gradient(_, __):
                """Get the gradient of a static feature."""

                return zeros((0,))

            # Map the feature types to the get functions
            get_functions = {
                'Dosiomics': get_dosiomic_gradient,
                'Radiomics': get_radiomic_gradient,
                'Statics': get_static_gradient}

            # Run the specific get function to retrieve the feature gradient
            feature_gradient = (
                get_functions[self.feature_map[feature]['class']](
                    feature, self.feature_map[feature]['segment']))

            return pad(feature_gradient, self.inputs['paddings'][
                self.feature_map[feature]['segment']])

        # Run the computation function for all features in the feature map
        gradients = map(compute_feature_gradient, (*self.feature_map,))

        return vstack(tuple(gradients))
