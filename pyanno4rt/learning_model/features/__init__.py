"""
Features module.

==================================================================

The module aims to provide methods and classes to handle the dosiomic and \
radiomic input features, including feature definitions and (re)calculation.
"""

# Author: Tim Ortkamp

# Import the abstract feature classes
from ._feature_class import DosiomicFeature, RadiomicFeature

# Import the dosiomic features
from ._dose_mean import DoseMean
from ._dose_deviation import DoseDeviation
from ._dose_maximum import DoseMaximum
from ._dose_minimum import DoseMinimum
from ._dose_skewness import DoseSkewness
from ._dose_kurtosis import DoseKurtosis
from ._dose_entropy import DoseEntropy
from ._dose_energy import DoseEnergy
from ._dose_nvoxels import DoseNVoxels
from ._dose_dx import DoseDx
from ._dose_vx import DoseVx
from ._dose_subvolume import DoseSubvolume
from ._dose_gradient import DoseGradient
from ._dose_moment import DoseMoment

# Import the radiomic features
from ._segment_area import SegmentArea
from ._segment_volume import SegmentVolume
from ._segment_eigenvalues import SegmentEigenvalues
from ._segment_eccentricity import SegmentEccentricity
from ._segment_compactness import SegmentCompactness
from ._segment_density import SegmentDensity
from ._segment_sphericity import SegmentSphericity
from ._segment_eigenmin import SegmentEigenmin
from ._segment_eigenmid import SegmentEigenmid
from ._segment_eigenmax import SegmentEigenmax

# Import the feature calculator
from ._feature_calculator import FeatureCalculator

# Import the feature map
from ._feature_map import feature_map

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
    'FeatureCalculator',
    'feature_map']
