"""Learning maps."""

# Author: Tim Ortkamp

# %% External package import

from tensorflow.keras.losses import (
    BinaryCrossentropy, BinaryFocalCrossentropy, KLDivergence)
from tensorflow.keras.optimizers import Adam, Ftrl, SGD

# %% Internal package import

from pyanno4rt.learning.features import (
    DoseDeviation, DoseDx, DoseEnergy, DoseEntropy, DoseGradient, DoseKurtosis,
    DoseMaximum, DoseMean, DoseMinimum, DoseMoment, DoseNVoxels, DoseSkewness,
    DoseSubvolume, DoseVx, SegmentArea, SegmentCompactness, SegmentDensity,
    SegmentEccentricity, SegmentEigenmax, SegmentEigenmid, SegmentEigenmin,
    SegmentEigenvalues, SegmentSphericity, SegmentVolume)
from pyanno4rt.learning.features import DynamicFeature, Label, StaticFeature
from pyanno4rt.learning.losses import brier_loss, log_loss
from pyanno4rt.learning.preprocessing import (
    Identity, StandardScaler, Whitening)

# %% Map definitions


COLUMNS = {
    'Dynamic Feature': DynamicFeature,
    'Label': Label,
    'StaticFeature': StaticFeature}

FEATURES = {
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

LOSSES = {
    'Brier score': brier_loss,
    'Logloss': log_loss}

NN_LOSSES = {
    'BCE': BinaryCrossentropy,
    'FocalBCE': BinaryFocalCrossentropy,
    'KLD': KLDivergence}

NN_OPTS = {
    'Adam': Adam,
    'Ftrl': Ftrl,
    'SGD': SGD}

TRANSFORMERS = {
    'Identity': Identity,
    'StandardScaler': StandardScaler,
    'Whitening': Whitening}
