"""Feature value and gradient (re)calculation."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from numpy import (
    array, array_equal, empty, fromiter, unravel_index, vstack, zeros)
from scipy.ndimage import zoom
from scipy.sparse import lil_matrix
from scipy.sparse import vstack as svstack

# %% Internal package import

from pyanno4rt.datahub import Datahub

# %% Class definition


class FeatureCalculator():
    """
    Feature value and gradient (re)calculation class.

    Parameters
    ----------
    write_features : bool
        Indicator for tracking the feature values.

    Attributes
    ----------
    write_features : bool
        See 'Parameters'.

    feature_history : ndarray or None
        Feature values per iteration. If ``write_features`` is False, this \
        attribute is not set.

    gradient_history : list or None
        Gradient matrices per iteration. If ``write_gradients`` is False, \
        this attribute is not set.

    radiomics : dict
        Dictionary for mapping the radiomic feature names to the radiomic \
        feature values. It allows to retrieve the feature values after first \
        computation and thus prevents unnecessary recalculation.

    demographics : dict
        Dictionary for mapping the demographic feature names to the \
        demographic feature values. It allows to retrieve the feature values \
        after first computation and thus prevents unnecessary recalculation.

    feature_inputs : dict
        Dictionary for collecting the candidate feature input values. This \
        allows to centralize the input retrieval for all calculations.

    __iteration__ : list
        Iteration numbers for the feature calculation and the optimization \
        problem. By keeping the two elements the same, it is assured that the \
        feature calculator is only active for new problem iterations, rather \
        than per evaluation step.

    __dose_cache__ : ndarray
        Cache array for the dose values.

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

        # Initialize the radiomics/demographics dictionaries to store values
        self.radiomics = {}
        self.demographics = {}

        # Initialize the feature map and history
        self.feature_map = None
        self.feature_history = None

        # Initialize the feature input dictionary
        self.feature_inputs = {'dose': None,
                               'dose_cube': None,
                               'require_cube': ('doseSubvolume',
                                                'doseMoment',
                                                'doseGradient'),
                               'require_spacing': ('doseGradient',),
                               'masks': None}

        # Initialize the iteration numbers for synchronization
        self.__iteration__ = [0, 0]

        # Initialize the dose and the feature cache
        self.__dose_cache__ = array([])
        self.__feature_cache__ = array([])

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

    def precompute(
            self,
            dose,
            segment):
        """
        Precompute the dose, dose cube and segment masks as inputs for the \
        feature calculation.

        Parameters
        ----------
        dose : tuple of ndarray
            Value of the dose for a single or multiple segments.

        segment : list of strings
            Names of the segments associated with the dose.
        """
        # Initialize the datahub
        hub = Datahub()

        # Get the information units from the datahub
        segmentation = hub.segmentation
        dose_information = hub.dose_information

        def precompute_dose(dose):
            """Precompute the dose."""
            return (dose_information['number_of_fractions'] * subdose.ravel()
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
                    order='F')] = self.feature_inputs['dose'][subsegment]

                return dose_cube

            return (get_subsegment_cube(subsegment) for subsegment in segment)

        def precompute_masks(segment):
            """Precompute the segment masks."""
            # Get the cube dimensions from the information unit
            ct_dimensions = hub.computed_tomography['cube_dimensions']
            dose_dimensions = dose_information['cube_dimensions']

            def get_subsegment_masks(subsegment):
                """Get the masks for a single segment."""
                # Initialize the radiomics mask by zeros
                radiomics_mask = zeros(ct_dimensions)

                # Insert ones at the indices of the segment
                radiomics_mask[unravel_index(
                    segmentation[subsegment]['raw_indices'],
                    ct_dimensions, order='F')] = 1

                # Get the dose mask by zooming into the radiomics mask
                dose_mask = zoom(radiomics_mask,
                                 (dose_dimensions[j]/ct_dimensions[j]
                                  for j in range(len(dose_dimensions))),
                                 order=0)

                return (radiomics_mask, dose_mask)

            return (get_subsegment_masks(subsegment) for subsegment in segment)

        # Add the precomputed dose by segment to the input dictionary
        self.feature_inputs['dose'] = dict(
            zip(segment, precompute_dose(dose)))

        # Add the precomputed dose cube by segment to the input dictionary
        self.feature_inputs['dose_cube'] = dict(
            zip(segment, precompute_dose_cube(segment)))

        # Check if the segment masks have not already been stored
        if (self.feature_inputs['masks'] is None
                or tuple(segment) != (*self.feature_inputs['masks'],)):

            # Add the precomputed masks by segment to the input dictionary
            self.feature_inputs['masks'] = dict(
                zip(segment, precompute_masks(segment)))

    def featurize(
            self,
            dose,
            segment,
            no_cache=False):
        """
        Convert dose and segment information into the feature vector.

        Parameters
        ----------
        dose : tuple of ndarray
            Value of the dose for a single or multiple segments.

        segment : list of strings
            Names of the segments associated with the dose.

        Returns
        -------
        ndarray
            Values of the calculated features.
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

        Parameters
        ----------
        dose : tuple of ndarray
            Value of the dose for a single or multiple segments.

        segment : list of strings
            Names of the segments associated with the dose.

        Returns
        -------
        ndarray
            Values of the calculated features.
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
                        for label in self.feature_inputs['require_cube']),
                    any(label in feature
                        for label in self.feature_inputs['require_spacing'])))

                # Compute the feature value if the boolean sum is zero or one
                if boolean_sum == 0:
                    return self.feature_map[feature]['computation'](
                        self.feature_inputs['dose'][segment])
                if boolean_sum == 1:
                    return self.feature_map[feature]['computation'](
                        self.feature_inputs['dose'][segment],
                        self.feature_inputs['dose_cube'][segment])

                # Otherwise, compute the value if the boolean sum is two
                return self.feature_map[feature]['computation'](
                    self.feature_inputs['dose'][segment],
                    self.feature_inputs['dose_cube'][segment],
                    fromiter(hub.dose_information['resolution'].values(),
                             float),
                    self.feature_inputs['masks'][segment][1])

            def get_radiomic_value(feature, segment):
                """Get the value of a radiomic feature."""
                # Check if the feature has already been computed
                if feature in self.radiomics:

                    # Return the feature value from the radiomics dictionary
                    return self.radiomics[feature]

                # Compute the feature value
                self.radiomics[feature] = self.feature_map[
                    feature]['computation'](
                        self.feature_inputs['masks'][segment][0],
                        fromiter(
                            hub.computed_tomography['resolution'].values(),
                            float))

                return self.radiomics[feature]

            def get_demographic_value(feature, _):
                """Get the value of a demographic feature."""
                # Map the demographic feature values to the feature names
                values = {'patientAge': None,
                          'patientSex': None,
                          'patientDaysafterrt': 150}

                # Check if the feature has already been computed
                if feature in self.demographics:

                    # Return the feature value from the demographics dictionary
                    return self.demographics[feature]

                # Compute the feature value
                self.demographics[feature] = self.feature_map[
                    feature]['computation'](values[feature])

                return self.demographics[feature]

            # Map the feature types to the get functions
            get_functions = {'Dosiomics': get_dosiomic_value,
                             'Radiomics': get_radiomic_value,
                             'Demographics': get_demographic_value}

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
            self.feature_history = vstack((self.feature_history,
                                           feature_vector))

        return feature_vector

    def gradientize(
            self,
            dose,
            segment):
        """
        Convert dose and segment information into the gradient matrix.

        Parameters
        ----------
        dose : tuple of ndarray
            Value of the dose for a single or multiple segments.

        segment : list of strings
            Names of the segments associated with the dose.

        Returns
        -------
        csr_matrix
            Matrix of the calculated gradients.
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
                # Determine the number of input conditions fulfilled
                boolean_sum = sum((
                    any(label in feature
                        for label in self.feature_inputs['require_cube']),
                    any(label in feature
                        for label in self.feature_inputs['require_spacing'])))

                # Compute the gradient if the boolean sum is zero or one
                if boolean_sum == 0:
                    return self.feature_map[feature]['differentiation'](
                        self.feature_inputs['dose'][segment],
                        dose_information['number_of_voxels'],
                        hub.segmentation[segment]['resized_indices'])
                if boolean_sum == 1:
                    return self.feature_map[feature]['differentiation'](
                        self.feature_inputs['dose'][segment],
                        self.feature_inputs['dose_cube'][segment])

                # Otherwise, compute the gradient if the boolean sum is two
                return self.feature_map[feature]['differentiation'](
                    self.feature_inputs['dose'][segment],
                    self.feature_inputs['dose_cube'][segment],
                    fromiter(dose_information['resolution'].values(), float),
                    self.feature_inputs['masks'][segment][1])

            def get_default_gradient(_, __):
                """Return a default gradient for non-dosiomic features."""
                return lil_matrix((1, dose_information['number_of_voxels']))

            # Map the feature types to the get functions
            get_functions = {'Dosiomics': get_dosiomic_gradient,
                             'Radiomics': get_default_gradient,
                             'Demographics': get_default_gradient}

            # Run the specific get function to retrieve the feature gradient
            feature_gradient = (
                get_functions[self.feature_map[feature]['class']](
                    feature, self.feature_map[feature]['segment']))

            return feature_gradient

        # Run the computation function for all features in the feature map
        gradients = map(compute_feature_gradient, (*self.feature_map,))

        return svstack(tuple(gradients))
