"""
Features module.

==================================================================

The module aims to provide methods and classes to handle the model input \
features, including feature definitions and the feature calculator.
"""

# Author: Tim Ortkamp

from pyanno4rt.learning.features._feature import DosiomicFeature, RadiomicFeature

from pyanno4rt.learning.features._dose_mean import DoseMean
from pyanno4rt.learning.features._dose_deviation import DoseDeviation
from pyanno4rt.learning.features._dose_maximum import DoseMaximum
from pyanno4rt.learning.features._dose_minimum import DoseMinimum
from pyanno4rt.learning.features._dose_skewness import DoseSkewness
from pyanno4rt.learning.features._dose_kurtosis import DoseKurtosis
from pyanno4rt.learning.features._dose_entropy import DoseEntropy
from pyanno4rt.learning.features._dose_energy import DoseEnergy
from pyanno4rt.learning.features._dose_nvoxels import DoseNVoxels
from pyanno4rt.learning.features._dose_dx import DoseDx
from pyanno4rt.learning.features._dose_vx import DoseVx
from pyanno4rt.learning.features._dose_subvolume import DoseSubvolume
from pyanno4rt.learning.features._dose_gradient import DoseGradient
from pyanno4rt.learning.features._dose_moment import DoseMoment

from pyanno4rt.learning.features._segment_area import SegmentArea
from pyanno4rt.learning.features._segment_volume import SegmentVolume
from pyanno4rt.learning.features._segment_eigenvalues import SegmentEigenvalues
from pyanno4rt.learning.features._segment_eccentricity import SegmentEccentricity
from pyanno4rt.learning.features._segment_compactness import SegmentCompactness
from pyanno4rt.learning.features._segment_density import SegmentDensity
from pyanno4rt.learning.features._segment_sphericity import SegmentSphericity
from pyanno4rt.learning.features._segment_eigenmin import SegmentEigenmin
from pyanno4rt.learning.features._segment_eigenmid import SegmentEigenmid
from pyanno4rt.learning.features._segment_eigenmax import SegmentEigenmax

from pyanno4rt.learning.features._columns import DynamicFeature, Label, StaticFeature
from pyanno4rt.learning.features._feature_calculator import FeatureCalculator

__all__ = [
    'DosiomicFeature',
    'RadiomicFeature',
    'DoseMean',
    'DoseDeviation',
    'DoseMaximum',
    'DoseMinimum',
    'DoseSkewness',
    'DoseKurtosis',
    'DoseEntropy',
    'DoseEnergy',
    'DoseNVoxels',
    'DoseDx',
    'DoseVx',
    'DoseSubvolume',
    'DoseGradient',
    'DoseMoment',
    'SegmentArea',
    'SegmentVolume',
    'SegmentEccentricity',
    'SegmentCompactness',
    'SegmentDensity',
    'SegmentSphericity',
    'SegmentEigenmin',
    'SegmentEigenmid',
    'SegmentEigenmax',
    'SegmentEigenvalues',
    'DynamicFeature',
    'Label',
    'StaticFeature',
    'FeatureCalculator']
