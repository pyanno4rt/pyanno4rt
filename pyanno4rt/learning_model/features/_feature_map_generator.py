"""Feature map generation."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from functools import partial
from inspect import isclass
from operator import methodcaller
from numpy import argmax
from fuzzywuzzy import fuzz

# %% Internal package import

from pyanno4rt.datahub import Datahub
import pyanno4rt.learning_model.features.catalogue as feature_catalogue
from pyanno4rt.tools import identity

# %% Class definition


class FeatureMapGenerator():
    """
    Feature map generation class.

    This class provides a mapping between the features from the data set, the \
    structures from the segmentation, and the definitions from the feature \
    catalogue. Matching is based on fuzzy or exact string matching.

    Parameters
    ----------
    model_label : str
        Label for the machine learning model.

    fuzzy_matching : bool
        Indicator for the use of fuzzy string matching (if 'False', exact \
        string matching is applied).

    Attributes
    ----------
    model_label : str
        See 'Parameters'.

    fuzzy_matching : bool
        See 'Parameters'.

    Notes
    -----
    String matching works best if (1) the segment names in the segmentation \
    dictionary do not include any special characters except "_" (which will \
    be removed before matching), and (2) the feature names follow the scheme \
    `<name of the segment>_<name of the feature in the catalogue>_\
    <optional parameters>`, e.g. "parotidLeft_doseMean" (mean dose to the \
    left parotid) or "parotidRight_doseGradient_x" (dose gradient in \
    x-direction for the right parotid).
    """

    def __init__(
            self,
            model_label,
            fuzzy_matching):

        # Initialize the datahub
        hub = Datahub()

        # Log a message about the initialization of the class
        hub.logger.display_info("Initializing feature map generator ...")

        # Get the instance attributes from the arguments
        self.model_label = model_label
        self.fuzzy_matching = fuzzy_matching

    def generate(
            self,
            feature_names):
        """
        Generate the feature map by string matching.

        Parameters
        ----------
        feature_names : list
            Names of the input features.

        Returns
        -------
        feature_map : dict
            Dictionary with information on the mapping of the input features \
            from the dataset with the segmented structures and their \
            computation/differentiation functions.
        """

        # Initialize the datahub
        hub = Datahub()

        # Log a message about the string matching
        hub.logger.display_info(
            f"Performing {'fuzzy' if self.fuzzy_matching else 'exact'} string "
            "matching for segments and feature definitions ...")

        def get_segment_from_cache(_, feature_segment):
            """Get the segment from the mapping cache."""

            return mapping_cache[feature_segment]

        def get_segment(feature_name, feature_segment):
            """Get the segment by string matching."""

            # Calculate the fuzzy partial ratios (similarity scores)
            scores = tuple(
                fuzz.partial_ratio(
                    feature_segment.lower(), segment.replace('_', '').lower())
                for segment in segments)

            # Get the index of the maximum score
            max_index = argmax(scores)

            # Check if fuzzy or exact string matching apply
            if self.fuzzy_matching or scores[max_index] == 100:

                # Get the matching segment
                segment_match = (segments[max_index],)

                # Add the matching segment to the mapping cache
                mapping_cache[feature_segment] = segment_match

                # Return the matching segment
                return segment_match

            # Log a message about a failed match
            hub.logger.display_error(
                f"No matching segment found for {feature_segment} in "
                f"{feature_name} ...")

            # Return None for the matching segment
            return None

        def get_definition_from_cache(
                _, feature_definition, feature_parameters):
            """Get the feature definition from the mapping cache."""

            return mapping_cache['_'.join(
                filter(None, (feature_definition, feature_parameters)))]

        def get_definition(
                feature_name, feature_definition, feature_parameters):
            """Get the feature definition by string matching."""

            # Calculate the fuzzy partial ratios (similarity scores)
            scores = tuple(
                fuzz.ratio(feature_definition.lower(), definition.lower())
                for definition in catalogue)

            # Get the index of the maximum score
            max_index = argmax(scores)

            # Check if fuzzy or exact string matching apply
            if self.fuzzy_matching or scores[max_index] == 100:

                # Get the matching definition name
                def_match = catalogue[max_index]

                # Get the definition class
                def_class = getattr(feature_catalogue, def_match).feature_class

                # Get the definition computation function
                def_computation = methods[feature_parameters is None](
                    getattr(feature_catalogue, def_match).compute,
                    feature_parameters)

                # Check if the definition class is 'Dosiomics'
                if def_class == 'Dosiomics':

                    # Get the definition differentiation function
                    def_differentiation = methods[feature_parameters is None](
                        getattr(feature_catalogue, def_match).differentiate,
                        feature_parameters)

                else:

                    # Set the definition differentiation function to None
                    def_differentiation = None

                # Get the matching full definition
                full_definition_match = (
                    def_class, def_computation, def_differentiation)

                # Add the matching full definition to the mapping cache
                mapping_cache['_'.join(filter(
                    None, (feature_definition, feature_parameters)))] = (
                        full_definition_match)

                # Return the matching full definition
                return full_definition_match

            # Log a message about a failed match
            hub.logger.display_error(
                f"No matching definition found for {feature_definition} in "
                f"{feature_name} ...")

            # Return None for the matching full definition
            return None

        # Get the segmented structures
        segments = (*hub.segmentation,)

        # Get the definition classes from the feature catalogue
        catalogue = tuple(
            definition for definition in dir(feature_catalogue)
            if isclass(getattr(feature_catalogue, definition)))

        # Initialize the feature mapping cache
        mapping_cache = {}

        # Split the feature names
        feature_name_splits = tuple(
            (split[0], split[1], None) if len(split) == 2
            else (split[0], split[1], split[2]) if len(split) == 3
            else (None, None, None)
            for split in map(methodcaller('split', '_'), feature_names))

        # Create a boolean mapping to the subfunctions
        get_segment_functions = {
            True: get_segment_from_cache, False: get_segment}
        get_definition_functions = {
            True: get_definition_from_cache, False: get_definition}

        # Create a boolean mapping to the internal functions
        methods = {True: identity, False: partial}

        # Set the keys for the feature submaps
        keys = ('segment', 'class', 'computation', 'differentiation')

        # Get the values for the feature submaps
        feature_matches = (
            get_segment_functions[split[0] in mapping_cache](name, split[0])
            + get_definition_functions[
                '_'.join(filter(None, (split[1], split[2]))) in mapping_cache]
            (name, split[1], split[2])
            for name, split in zip(feature_names, feature_name_splits))

        # Merge the keys and the values into the feature map
        feature_map = {
            feature_name: {key: value for key, value in zip(keys, matches)
                           if value is not None}
            for feature_name, matches in zip(feature_names, feature_matches)}

        # Enter the feature map into the datahub
        hub.feature_maps[self.model_label] = feature_map

        return feature_map
