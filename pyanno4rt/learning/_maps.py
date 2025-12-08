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
from pyanno4rt.learning.preprocessing import StandardScaler, Whitening
from pyanno4rt.learning.tuning import (
    BayesHPTuner, GridHPTuner, RandomHPTuner, TuneSpaceDT, TuneSpaceKNN,
    TuneSpaceLR, TuneSpaceNB, TuneSpaceNN, TuneSpaceRF, TuneSpaceSVM)
from pyanno4rt.learning.losses import auc_loss, brier_loss, log_loss

# %% Map definitions


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
    'AUC': auc_loss,
    'Brier score': brier_loss,
    'Logloss': log_loss}

NETWORK_LOSSES = {
    'BCE': BinaryCrossentropy,
    'FocalBCE': BinaryFocalCrossentropy,
    'KLD': KLDivergence}

NETWORK_OPTIMIZERS = {
    'Adam': Adam,
    'Ftrl': Ftrl,
    'SGD': SGD}

STEPS = {
    'Identity': 0,
    'StandardScaler': 1,
    'Whitening': 2}

TUNERS = {
    'Bayes': BayesHPTuner,
    'Grid': GridHPTuner,
    'Random': RandomHPTuner}

TUNE_SPACES = {
    'Decision Tree': TuneSpaceDT,
    'K-Nearest Neighbors': TuneSpaceKNN,
    'Logistic Regression': TuneSpaceLR,
    'Naive Bayes': TuneSpaceNB,
    'Neural Network': TuneSpaceNN,
    'Random Forest': TuneSpaceRF,
    'Support Vector Machine': TuneSpaceSVM}

TRANSFORMERS = {
    'StandardScaler': StandardScaler,
    'Whitening': Whitening}

