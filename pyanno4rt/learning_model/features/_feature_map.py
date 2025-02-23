"""Features map."""

# Author: Tim Ortkamp

# %% Internal package import

from pyanno4rt.learning_model.features import (
    DoseDeviation, DoseDx, DoseEnergy, DoseEntropy, DoseGradient, DoseKurtosis,
    DoseMaximum, DoseMean, DoseMinimum, DoseMoment, DoseNVoxels, DoseSkewness,
    DoseSubvolume, DoseVx, SegmentArea, SegmentCompactness, SegmentDensity,
    SegmentEccentricity, SegmentEigenmax, SegmentEigenmid, SegmentEigenmin,
    SegmentEigenvalues, SegmentSphericity, SegmentVolume)

# %% Map definition


feature_map = {
    'Dose Deviation': DoseDeviation,
    'Dx': DoseDx,
    'Dose Energy': DoseEnergy,
    'Dose Entropy': DoseEntropy,
    'Dose Gradient': DoseGradient,
    'Dose Kurtosis': DoseKurtosis,
    'Dose Maximum': DoseMaximum,
    'Dose Mean': DoseMean,
    'Dose Minimum': DoseMinimum,
    'Dose Moment': DoseMoment,
    'Dose Voxels': DoseNVoxels,
    'Dose Skewness': DoseSkewness,
    'Dose Subvolume': DoseSubvolume,
    'Vx': DoseVx,
    'Segment Area': SegmentArea,
    'Segment Compactness': SegmentCompactness,
    'Segment Density': SegmentDensity,
    'Segment Eccentricity': SegmentEccentricity,
    'Segment Eigenmax': SegmentEigenmax,
    'Segment Eigenmid': SegmentEigenmid,
    'Segment Eigenmin': SegmentEigenmin,
    'Segment Eigenvalues': SegmentEigenvalues,
    'Segment Sphericity': SegmentSphericity,
    'Segment Volume': SegmentVolume}
