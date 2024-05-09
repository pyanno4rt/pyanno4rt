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
    fuzzy_matching : bool
        Indicator for the use of fuzzy string matching (if 'False', exact \
        string matching is applied).

    Attributes
    ----------
    fuzzy_matching : bool
        See 'Parameters'.

    feature_map : dict
        Dictionary with information on the mapping of features in the dataset \
        with the segmented structures and their computation/differentiation \
        functions.

    Notes
    -----
    In the current implementation, string matching works best if:
        - names from segments in the segmentation do not have any special \
          characters except "_" (which will automatically be removed before \
          matching);
        - feature names follow the scheme `<name of the segment>\
          _<name of the feature in the catalogue>_<optional parameters>`, \
          e.g. "parotidLeft_doseMean" (mean dose to the left parotid) or \
          "parotidRight_doseGradient_x" (dose gradient in x-direction for the \
          right parotid).
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
            data_information):
        """
        Generate the feature map by fuzzy or exact string matching.

        Parameters
        ----------
        ...

        Returns
        -------
        feature_map : dict
            Dictionary with information on the mapping of features in the \
            dataset with the segmented structures and their \
            computation/differentiation functions.
        """
        # Initialize the datahub
        hub = Datahub()

        # Log a message about the string matching
        hub.logger.display_info("Performing {} string matching for segments "
                                "and feature definitions ..."
                                .format('fuzzy' if self.fuzzy_matching
                                        else 'exact'))

        def get_segment_from_cache(_, feature_segment):
            """Get the segment from the mapping cache."""
            return mapping_cache[feature_segment]

        def get_segment(feature_name, feature_segment):
            """Get the segment by fuzzy or exact string matching."""
            # Calculate the fuzzy partial ratios (similarity scores)
            scores = tuple(fuzz.partial_ratio(feature_segment.lower(),
                                              segment.replace('_', '').lower())
                           for segment in segments)

            # Get the index of the maximum score
            max_index = argmax(scores)

            # Check if fuzzy matching is used or the score is 100 (exact match)
            if self.fuzzy_matching or scores[max_index] == 100:

                # Get the matching segment
                segment_match = [segments[max_index]]

                # Add the matching segment to the mapping cache
                mapping_cache[feature_segment] = segment_match

                return segment_match

            # Log a message about an error and return None if no match is found
            hub.logger.display_error("No valid segment match detected for {} "
                                     "in {} ..."
                                     .format(feature_segment, feature_name))

            return None

        def get_definition_from_cache(_, feature_definition,
                                      feature_parameters):
            """Get the feature definition from the mapping cache."""
            return mapping_cache['_'.join(
                filter(None, (feature_definition, feature_parameters)))]

        def get_definition(feature_name, feature_definition,
                           feature_parameters):
            """Get the feature definition by fuzzy or exact string matching."""
            # Calculate the fuzzy partial ratios (similarity scores)
            scores = [fuzz.ratio(feature_definition.lower(), clf.lower())
                      for clf in catalogue]

            # Get the index of the maximum score
            max_index = argmax(scores)

            # Check if fuzzy matching is used or the score is 100 (exact match)
            if self.fuzzy_matching or scores[max_index] == 100:

                # Get the matching feature definition from the catalogue
                catalogue_match = catalogue[max_index]

                # Get the feature class from the catalogue match
                feature_class = getattr(
                    feature_catalogue, catalogue_match).feature_class

                # Get the computation function from the catalogue match
                feature_computation = methods[feature_parameters is None](
                        getattr(feature_catalogue, catalogue_match).compute,
                        feature_parameters)

                # Check if the feature class is 'Dosiomics'
                if feature_class == 'Dosiomics':

                    # Get the differentiation function from the catalogue match
                    feature_differentiation = methods[
                        feature_parameters is None](
                            getattr(feature_catalogue,
                                    catalogue_match).differentiate,
                            feature_parameters)

                else:

                    # Set the differentiation function to None
                    feature_differentiation = None

                # Connect the matching values to get the definition match
                definition_match = list((feature_class, feature_computation,
                                         feature_differentiation))

                # Add the definition match to the mapping cache
                mapping_cache['_'.join(
                    filter(None, (feature_definition,
                                  feature_parameters)))] = definition_match

                return definition_match

            # Log a message about an error and return None if no match is found
            hub.logger.dispError("\t\t\t No valid function match detected "
                                 "for {} in {} ..."
                                 .format(feature_definition, feature_name))

            return None

        # Get the structures from the segmentation
        segments = (*hub.segmentation,)

        # Get the classes from the feature catalogue
        catalogue = tuple(definition for definition in dir(feature_catalogue)
                          if isclass(getattr(feature_catalogue, definition)))

        # Split the feature names into its components
        feature_name_splits = tuple(
            (None, split[0], None) if len(split) == 1
            else (split[0], split[1], None) if len(split) == 2
            else (split[0], split[1], split[2])
            for split in map(methodcaller('split', '_'),
                             data_information['feature_names']))

        # Decompose the splits into segments, definitions and parameters
        feature_segments, feature_definitions, feature_parameters = zip(
            *feature_name_splits)

        # Map the caching indicator to the get functions
        get_segment_functions = {True: get_segment_from_cache,
                                 False: get_segment}
        get_definition_functions = {True: get_definition_from_cache,
                                    False: get_definition}

        # Map an indicator for the absence of feature parameters to the \
        # argument methods
        methods = {True: identity, False: partial}

        # Initialize the feature mapping cache
        mapping_cache = {}

        # Set the labels (keys) for the feature map
        labels = ('segment', 'class', 'computation', 'differentiation')

        # Get the matches (values) for the feature map
        matches = ((get_segment_functions[feature_segments[i] in mapping_cache]
                    (data_information['feature_names'][i],
                     feature_segments[i])
                    if feature_segments[i] is not None else [None])
                   + get_definition_functions[
                       '_'.join(filter(None, (feature_definitions[i],
                                              feature_parameters[i])))
                       in mapping_cache](
                           data_information['feature_names'][i],
                           feature_definitions[i],
                           feature_parameters[i])
                   for i in range(len(data_information['feature_names'])))

        # Construct the output feature map
        feature_map = {feature_name: {label: value for label, value in zip(
            labels, match) if value is not None}
            for feature_name, match in zip(data_information['feature_names'],
                                           matches)}

        # Enter the feature map into the datahub
        hub.feature_maps[self.model_label] = feature_map

        return feature_map
