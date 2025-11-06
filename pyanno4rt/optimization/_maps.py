"""Optimization maps."""

# Author: Tim Ortkamp

# %% Internal package import

from pyanno4rt.optimization.components import (
    DecisionTreeOutcome, DoseUniformity, EquivalentUniformDose,
    KNeighborsOutcome, LogisticRegressionOutcome, LQPoissonTCP,
    LymanKutcherBurmanNTCP, MaximumDVH, MeanDose, MinimumDVH,
    NaiveBayesOutcome, NeuralNetworkOutcome, RandomForestOutcome,
    SquaredDeviation, SquaredOverdosing, SquaredUnderdosing,
    SupportVectorMachineOutcome)
from pyanno4rt.optimization.initializers import (
    DataMedoidInitializer, TargetCoverageInitializer, WarmStartInitializer)
from pyanno4rt.optimization.methods.lexicographic import LexicographicOptimization
from pyanno4rt.optimization.methods.pareto import ParetoOptimization
from pyanno4rt.optimization.methods.weighted import WeightedSumOptimization
from pyanno4rt.optimization.projections import (
    ConstantRBEProjection, DoseProjection)
from pyanno4rt.optimization.solvers import (
    IpyoptSolver, PymooSolver, PyPop7Solver, SciPySolver)

# %% Map definitions


COMPONENTS = {
    'Decision Tree Outcome': DecisionTreeOutcome,
    'Dose Uniformity': DoseUniformity,
    'Equivalent Uniform Dose': EquivalentUniformDose,
    'K-Nearest Neighbors Outcome': KNeighborsOutcome,
    'Logistic Regression Outcome': LogisticRegressionOutcome,
    'LQ Poisson TCP': LQPoissonTCP,
    'Lyman-Kutcher-Burman NTCP': LymanKutcherBurmanNTCP,
    'Maximum DVH': MaximumDVH,
    'Mean Dose': MeanDose,
    'Minimum DVH': MinimumDVH,
    'Naive Bayes Outcome': NaiveBayesOutcome,
    'Neural Network Outcome': NeuralNetworkOutcome,
    'Random Forest Outcome': RandomForestOutcome,
    'Squared Deviation': SquaredDeviation,
    'Squared Overdosing': SquaredOverdosing,
    'Squared Underdosing': SquaredUnderdosing,
    'Support Vector Machine Outcome': SupportVectorMachineOutcome}

INITIALIZERS = {
    'data-medoid': DataMedoidInitializer,
    'target-coverage': TargetCoverageInitializer,
    'warm-start': WarmStartInitializer}

METHODS = {
    'lexicographic': LexicographicOptimization,
    'pareto': ParetoOptimization,
    'weighted-sum': WeightedSumOptimization}

PROJECTIONS = {
    'photon': DoseProjection,
    'proton': ConstantRBEProjection}

SOLVERS = {
    'ipyopt': IpyoptSolver,
    'pymoo': PymooSolver,
    'pypop7': PyPop7Solver,
    'scipy': SciPySolver}
